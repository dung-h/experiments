#!/usr/bin/env python3
"""Render thesis synthesis from saved evidence only; never load a fitted model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/thesis_synthesis/v1"
CLOSURE = "artifacts/final_rq_closure_v2"
PINS = {}
NODE = "/home/server/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
SOURCES = {
    "closure": f"{CLOSURE}/README.md",
    "claims": f"{CLOSURE}/final_claim_matrix.csv",
    "rq": f"{CLOSURE}/final_rq_status.csv",
    "contributions": f"{CLOSURE}/thesis_contributions.md",
    "gaps": f"{CLOSURE}/final_gap_register.md",
    "verdict": f"{CLOSURE}/experiment_completion_decision.md",
    "provenance": f"{CLOSURE}/provenance_resolution.md",
    "qpu": "artifacts/benchmark_deep_dive/tables/qpu_metrics_v1.csv",
    "qpu_pairs": "artifacts/real_qpu/qualified_logical_panel/summary/shared_row_comparisons.csv",
    "controls": "artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv",
    "taxonomy": "artifacts/thesis_interpretation/v1/tables/estimator_identity.csv",
    "registry": "artifacts/thesis_interpretation/v1/estimator_fidelity_registry.md",
    "generalization": "artifacts/thesis_interpretation/v1/tables/generalization_robustness.csv",
    "grid": "artifacts/final_research/rq4_report.md",
    "sim": "artifacts/thesis_interpretation/v1/tables/track_b_findings.csv",
    "sim_pairs": "artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv",
    "mps": "artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv",
    "tn": "artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv",
    "pasqal": "artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv",
    "p4": "artifacts/am1_phase4/v1/analysis.json",
    "p5": "artifacts/am1_phase5_dense_b2/v1/analysis.json",
    "dense_full": "artifacts/am1_phase5_dense_b2/v1/full_context_accuracy.csv",
    "feature": "artifacts/benchmark_interpretation/v1/mali_feature_schema_audit.md",
    "recovery": "artifacts/am1_preprocessing_recovery/v1/root_cause_analysis.md",
    "replay": "artifacts/am1_preprocessing_recovery/v1/validation_report.md",
    "qcec": "artifacts/real_qpu/provenance_equivalence/report.md",
    "crosswalk": "artifacts/thesis_interpretation/v1/evidence_matrix_crosswalk.csv",
    "contract": "docs/research_contract.md",
    "current": "docs/research_conclusions.md",
    "exhaustive": "artifacts/final_exhaustive_validation/v1/validation_report.md",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pin(path):
    p = Path(path)
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    h = sha(p)
    assert str(p) not in PINS or PINS[str(p)] == h, f"source changed: {p}"
    PINS[str(p)] = h
    return p

def read(path):
    return pin(path).read_text(encoding="utf-8")

def rows(path):
    return list(csv.DictReader(io.StringIO(read(path))))

def doc(path):
    return json.loads(read(path))

def refs(*keys):
    return "\n\nNguồn: " + "; ".join(f"[{key}](../../../{SOURCES[key]})" for key in keys) + ".\n"

def cited(row, *keys):
    return {**row, "source_paths": ";".join(SOURCES[k] for k in keys),
            "source_sha256": ";".join(sha(pin(SOURCES[k])) for k in keys)}

def numeric(value):
    if value == "": return ""
    if re.fullmatch(r"-?\d+", value): return int(value)
    try: return float(value)
    except ValueError: return value

def polish_prose(content):
    """Space prose quantities without changing identifiers, paths, tables or code."""
    keep = {'RQ', 'B', 'FP', 'T', 'U', 'H', 'D', 'C', 'AM', 'p', 'r', 'c', 'chi', 'Hyb'}
    def spaced(match):
        word, number = match.groups()
        return word+number if word in keep else word+' '+number
    result=[]
    code=False
    for line in content.splitlines(keepends=True):
        if line.startswith('```'): code=not code
        if code or line.startswith('|') or '`' in line or '](' in line:
            result.append(line); continue
        line=re.sub(r'([A-Za-zÀ-ỹ_]+)(\d+)',spaced,line)
        line=re.sub(r'(?<=\d)(?=(?:observations|groups|rows|globals|cases|positive|independent)\b)',' ',line)
        line=re.sub(r',(?=[A-Za-zÀ-ỹ])',', ',line)
        result.append(line)
    return ''.join(result)

def markdown_table(data, columns):
    def value(v):
        if isinstance(v, float): return f"{v:.6g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n" + "\n".join(
        "| " + " | ".join(value(r.get(c, "")) for c in columns) + " |" for r in data) + "\n"

def build():
    # Carry forward the complete closure source-identity checks, not its old PASS alone.
    validation = read(f"{CLOSURE}/validation_report.md")
    hash_section = validation.split("## Captured source hashes", 1)[1]
    for line in hash_section.splitlines():
        match = re.fullmatch(r"\| (.+) \| ([a-f0-9]{64}) \|", line)
        if match:
            assert sha(pin(match[1])) == match[2], f"closure source drift: {match[1]}"
    for p in (ROOT/CLOSURE).iterdir():
        if p.is_file(): pin(p)
    pin(__file__)
    pin("benchmark_v1/scripts/export_thesis_tables.mjs")
    for p in SOURCES.values(): pin(p)
    for p in ["AGENTS.md", "artifacts/release_review/scientific_freeze.json",
              "artifacts/final_research/manifest.json", "artifacts/am1_phase4/v1/accuracy_cost_tradeoff.md",
              "artifacts/am1_phase5_dense_b2/v1/accuracy_cost_tradeoff.md",
              "artifacts/am1_preprocessing_recovery/v1/historical_environment_provenance.md"]: pin(p)
    qpu = {r["method_id"]: r for r in rows(SOURCES["qpu"])}
    sim = rows(SOURCES["sim"])
    p4, p5 = doc(SOURCES["p4"]), doc(SOURCES["p5"])
    rq = rows(SOURCES["rq"])
    assert len(rq) == 10 and all(r["blocking"] == "NON-BLOCKING" for r in rq)
    assert "210" in read(SOURCES["replay"]) and '"independently_encoded_model_ready_exact": 210' in read(SOURCES["replay"])
    mps = rows(SOURCES["mps"])
    selected_mps = [r for r in mps if r["context_id"] == "cudaq_mps_ladder_c44"]
    assert len(selected_mps)==2 and all(r["selected_observed"]=="140" and r["quality_violations"]=="64" for r in selected_mps)
    tables = {}
    taxonomy = []
    for r in rows(SOURCES["taxonomy"]):
        if r["card_id"] in {"historical_controls", "holdout_families"}: continue
        track = "A/B" if r["card_id"] in {"references", "service_bridge"} else "B" if r["card_id"] in {"sim_mali","azizov","sim_classical","family_joint","family_fixed","maestro","tn","pasqal"} else "A"
        taxonomy.append(cited(dict(method=r["reader_label"], track=track, original_target=r["original_target"],
            benchmark_target=r["benchmark_target"], inputs=r["implemented_inputs"],
            method_category=r["original_mechanism"], fidelity_classification=r["category"],
            major_caveat=r["adaptation"]+". "+r["prohibited_claim"]),"taxonomy","registry"))
    tables["a_method_taxonomy.csv"] = taxonomy
    methods = ["mali_logical_graph","mali_global_mlp","qonductor_budgeted_selector","qonductor_extra_trees",
        "qonductor_random_forest","qonductor_gradient_boosting","qonductor_adaboost",
        "qonductor_histogram_gradient_boosting","qonductor_polynomial_regression","outer_train_median","outer_train_source_median"]
    main = []
    for method in methods:
        r=qpu[method]; assert r["n"] == "4515"
        main.append(cited(dict(method=method, n=int(r["n"]), coverage=float(r["coverage"]),
            mae_s=float(r["mae_seconds"]), medae_s=float(r["medae_seconds"]), rmse_s=float(r["rmse_seconds"]),
            r2=float(r["r2_seconds"]), p99_absolute_error_s=float(r["p99_ae_seconds"]),
            source_balanced_mae_s=float(r["source_balanced_mae"]), equal_group_mae_s=float(r["equal_group_mae"]),
            evaluation_target="source-retained observed execution/service time",
            comparison_scope="shared4515 OOF; not IID structures or equal-information ablation"),"qpu","qpu_pairs"))
    tables["b_qpu_main_accuracy.csv"] = main
    general = []
    general_fields=["case_id","experiment","method_id","n","mae_seconds","medae_seconds","rmse_seconds","r2_seconds",
        "structural_groups","known_group_test_rows","new_group_test_rows","exact_consumed_input_collision_rows",
        "train_global_median_mae_seconds","selector_mae_seconds","graph_minus_mlp_observed_mae_delta_seconds",
        "graph_minus_mlp_ci_low_seconds","graph_minus_mlp_ci_high_seconds","uncertainty_status"]
    for r in rows(SOURCES["generalization"]):
        general.append(cited({**{k:numeric(r[k]) for k in general_fields},
            "interpretation":"per-case median-of-three; conditional on fitted pipelines; case assignments overlap across cases"},"generalization","grid"))
    assert len(general)==42 and len({r['case_id'] for r in general})==21
    tables["c_qpu_generalization.csv"] = general
    chosen = {
        "cudaq_dense_statevector_fp32":{"random_forest","mali_style_dense_graph","matched_metadata_mlp"},
        "cudaq_dense_statevector_fp64":{"random_forest","mali_style_dense_graph","matched_metadata_mlp"},
        "qiskit_aer_fake_sherbrooke_noisy_opt1":{"azizov_gnn_source_median_three_seeds","azizov_gnn_transpiled_median_three_seeds","classical_xgboost_transpiled"},
        "cudaq_mps_fp64_bond16_warm_state_execution_seconds":{"graph_median_three_seeds","ridge_alpha_1"},
        "cudaq_mps_ladder_c44":{"family_conditioned","family_agnostic_ablation"},
        "cudaq_mps_ladder_family_holdout":{"family_conditioned","family_agnostic_ablation"},
        "qcsim_statevector_cpu_fp64_1000_shots":{"maestro_style_cpu_sv_component_optimizer_off_adaptation","nested_grouped_ridge"},
    }
    context_rows=[]
    for r in sim:
        if r["method_id"] not in chosen.get(r["context_id"],set()): continue
        context_rows.append(cited(dict(context=r["context_id"],method=r["method_id"],assigned_n=int(r["assigned_n"]),
            evaluated_n=int(r["n"]),coverage=float(r["coverage"]),mae_s=float(r["mae_seconds"]),r2=float(r["r2_seconds"]),
            runtime_boundary=r["runtime_boundary"],reported_quantity="OOF accuracy",unit="seconds",
            diagnostic_value="",major_caveat="same context/shared hashes or rungs only; not cross-engine ranking"),"sim","sim_pairs"))
    for r in rows(SOURCES["tn"]):
        if r["slice_type"] != "stratum": continue
        context_rows.append(cited(dict(context="cuTensorNet "+r["slice"],method="native RUNTIME_EST",assigned_n="not copied; see native plan attempt manifest",
            evaluated_n=int(r["session_rows"]),coverage="native successful-plan slice",mae_s="NOT_COMPARABLE",r2="NOT_REPORTED",
            runtime_boundary="same-plan warm scalar contraction; all-slice estimate",reported_quantity="median EST/actual ratio",unit="dimensionless",
            diagnostic_value=float(r["ratio_median"]),major_caveat="sessions != unique circuits; path/build not contraction clock"),"tn","registry"))
    pasqal=rows(SOURCES["pasqal"]); unique={r["cell"]:r for r in pasqal}
    assert len(unique)==18 and sum(r["quality_pass"]=="True" for r in unique.values())==17
    context_rows.append(cited(dict(context="Pasqal analog MPS/TDVP",method="local companion grid",assigned_n=18,evaluated_n=18,coverage=1,
        mae_s="NOT_A_PREDICTIVE_SCORE",r2="NOT_A_PREDICTIVE_SCORE",runtime_boundary="network/first/warm separately",
        reported_quantity="quality-pass count",unit="cells",diagnostic_value=17,
        major_caveat="analog workload; coefficients not promoted/identified"),"pasqal","registry"))
    tables["d_simulator_contexts.csv"] = context_rows
    cost=[]
    p4unc=p4["uncertainty"]["all_selected"]
    for r in p4["summaries"]:
        if r["source"]!="all_selected" or r["method"] not in {"graph","mlp"}:continue
        share=next(s['shares']['preparation'] for s in p4['shares'] if s['method']==('mali_logical_graph' if r['method']=='graph' else 'mali_global_mlp'))
        cost.append(cited(dict(context="QPU exact45",method=r['method'],n=45,accuracy_unit="s",mae=r['mae_seconds'],rmse=r['rmse_seconds'],r2=r['r2'],
            median_B2_ms=r['median_request_median_B2_ms'],preparation_fraction_accumulated_B2=share,
            paired_mae_graph_minus_mlp=p4['paired']['observed_mae_graph_minus_mlp_seconds'],
            paired_ci_low=p4unc['mae_difference_graph_minus_mlp_seconds'][0],paired_ci_high=p4unc['mae_difference_graph_minus_mlp_seconds'][1],
            marginal_median_B2_ratio_graph_over_mlp=p4['paired']['ratio_of_marginal_latency_medians'],
            ratio_ci_low=p4unc['marginal_median_latency_ratio'][0],ratio_ci_high=p4unc['marginal_median_latency_ratio'][1],
            population="38 Ma-Li+7 Qonductor; no QPack; not full4515",cost_boundary="warm full B2; model load/offline training excluded"),"p4"))
    for precision,block in p5['contexts'].items():
        for r in block['summary']:
            if r['method'] not in {'graph','mlp'}:continue
            share=next(s['shares']['preparation'] for s in p5['shares'] if s['context']=='dense_'+precision and s['method']==('mali_style_dense_graph' if r['method']=='graph' else 'matched_metadata_mlp'))
            cost.append(cited(dict(context="Dense "+precision.upper()+" exact30",method=r['method'],n=30,accuracy_unit="ms",mae=r['mae_seconds']*1000,
                rmse=r['rmse_seconds']*1000,r2=r['r2'],median_B2_ms=r['median_request_median_B2_ms'],preparation_fraction_accumulated_B2=share,
                paired_mae_graph_minus_mlp=block['paired']['observed_mae_graph_minus_mlp_seconds']*1000,
                paired_ci_low=block['uncertainty']['mae_difference_graph_minus_mlp_seconds'][0]*1000,
                paired_ci_high=block['uncertainty']['mae_difference_graph_minus_mlp_seconds'][1]*1000,
                marginal_median_B2_ratio_graph_over_mlp=block['paired']['ratio_of_marginal_latency_medians'],
                ratio_ci_low=block['uncertainty']['marginal_median_latency_ratio'][0],ratio_ci_high=block['uncertainty']['marginal_median_latency_ratio'][1],
                population="selected30 per precision; not full150",cost_boundary="warm full B2; historical simulator execution cross-environment"),"p5","dense_full"))
    assert len(cost)==6 and cost[0]['paired_ci_low']<0<cost[0]['paired_ci_high']
    assert all(r['paired_ci_low']>0 and r['ratio_ci_low']>1 for r in cost[2:])
    tables['e_accuracy_cost.csv']=cost
    failures=[
        ("Family-OOD overprediction","QNN26/Random30: cả Graph/MLP overpredict mọi test row; negative R²","QPU held-out family","Relative winner không đồng nghĩa practically useful prediction","generalization"),
        ("Representation collision","Kyoto148/148; Osaka149/192; Quito1051/1051 MLP fit-vector matches, no observation-ID leakage","QPU backend-context","Global-only context blind; không claim learned hardware adaptation","grid"),
        ("Ranking sensitivity","Pooled Graph<MLP; equal-group MLP<Graph; source-balanced selector<Graph","QPU shared4515","Estimand/source composition là phần của kết luận, không lựa chọn metric đẹp","qpu"),
        ("Tail sensitivity","p99/max error trong Table B; large negative R² ở QNN/Random","QPU OOF/holdouts","MAE đơn lẻ không mô tả tail risk","qpu"),
        ("Subset-selection sensitivity","Dense exact30 R² cao hơn full150 neural R² khoảng0.13–0.14","Dense FP32/FP64","Không biến timed-subset accuracy thành full-context performance","dense_full"),
        ("Metric dependence","Dense train-median MedAE tốt hơn neural, MAE/RMSE/R² kém hơn","Dense exact30","Không một metric/winner hợp với mọi mục đích","p5"),
        ("Unsafe quality selection","64/140 choices vi phạm gate ở mỗi frozen c44 selector;10/150 chưa assessed","MPS ladder c44","Runtime/classifier fit không đủ cho safe approximation choice","mps"),
        ("Native cost-model bias","Same-plan EST/actual median core3.734/frontier4.132","cuTensorNet","Cost estimate cần đánh giá đúng plan/clock; không suy ra opaque formula","tn"),
        ("Preprocessing incompatibility","38/45 QPU graph-order mismatch trước recovery;210/210 model-ready exact sau recovery","AM1 frozen input replay","Graph-semantic agreement không thay được exact consumed-input identity","replay"),
        ("Analog quality failure","17/18 cells pass; chi16 quench cell fails","Pasqal companion","Chạy hoàn tất không bảo đảm đủ quality; chưa validated fitted predictor","pasqal"),
    ]
    tables['f_failure_modes.csv']=[cited(dict(failure_mode=a,evidence=b,affected_context=c,scientific_implication=d),k) for a,b,c,d,k in failures]
    tables['g_rq_closure.csv']=[cited(dict(rq=r['rq'],track=r['track'],status=r['status'],blocking=r['blocking'],explanation=r['explanation']),"rq") for r in rq]
    gap_text=read(SOURCES['gaps']); gaps=[]
    for line in gap_text.splitlines():
        if not re.match(r'\| G\d+ \|',line):continue
        vals=[v.strip() for v in line.strip('|').split('|')]
        gaps.append(cited(dict(zip(['id','rq','track','gap','impact','blocking','future_work_value','execution_needed'],vals)),"gaps"))
    assert len(gaps)==9
    tables['h_limitations.csv']=gaps
    bodies=write_narratives(main, cost)
    # Compact Markdown views; CSV preserves precision and scoped detail, not huge registries.
    views={
        'a_method_taxonomy.csv':['method','track','original_target','benchmark_target','inputs','method_category','fidelity_classification','major_caveat'],
        'b_qpu_main_accuracy.csv':['method','n','mae_s','r2','source_balanced_mae_s','equal_group_mae_s','p99_absolute_error_s'],
        'c_qpu_generalization.csv':['case_id','method_id','n','structural_groups','mae_seconds','r2_seconds','exact_consumed_input_collision_rows'],
        'd_simulator_contexts.csv':['context','method','evaluated_n','mae_s','r2','runtime_boundary','reported_quantity','diagnostic_value'],
        'e_accuracy_cost.csv':['context','method','n','accuracy_unit','mae','rmse','r2','median_B2_ms'],
        'f_failure_modes.csv':['failure_mode','evidence','affected_context','scientific_implication'],
        'g_rq_closure.csv':['rq','track','status','blocking','explanation'],
        'h_limitations.csv':['id','rq','track','gap','blocking','impact'],
    }
    titles=['A — Method taxonomy','B — Track A main accuracy','C — Track A generalization','D — Track B by context',
            'E — Accuracy and warm full-B2 cost','F — Failure modes','G — RQ closure','H — Limitations/future work']
    canonical='# Canonical thesis tables\n\nCSV giữ precision từ saved evidence và source path/hash theo từng row. Markdown làm tròn để đọc, không tạo metric mới. Không đọc dọc Table D để tìm universal simulator winner.\n'
    for (name,data),title in zip(tables.items(),titles):
        visible=data
        if name=='c_qpu_generalization.csv':
            visible=[r for r in data if r['case_id'] in {'family:qnn','family:random','family:qft','backend_context:ibm_kyoto','backend_context:ibm_osaka','backend_context:ibm_quito','qpack_size:train_through_4','qpack_size:train_through_5'}]
        canonical+=f'\n## Table {title}\n\n[CSV](canonical_tables/{name}).\n\n'+markdown_table(visible,views[name])
        if name=='a_method_taxonomy.csv':canonical+='\nCategory là component/implementation, không mặc định toàn paper. Faithful reproduction là taxonomy chuẩn nhưng không gán nhãn đó cho Graph/MLP hiện tại.\n'
        if name=='b_qpu_main_accuracy.csv':canonical+='\nSource median nhận source identity nên không equal-information reference. Native analytical variants có eligible denominators khác; không trộn vào bảng full4515 này. Raw/calibrated details: [controls](../../../'+SOURCES['controls']+').\n'
        if name=='c_qpu_generalization.csv':canonical+='\nMarkdown chọn các case minh họa theo mục tiêu synthesis; CSV giữ đủ42 neural method rows/21 cases. Đây không phải chọn lại seed/model. CPU selectors và train-only references trong CSV; source-weighted parent sensitivity ở Table B.\n'
        if name=='e_accuracy_cost.csv':
            canonical+='\nMAE/RMSE và B2 là hai đại lượng khác nhau, không lấy hiệu. Paired CI và preparation fraction giữ trong CSV. Ratio là ratio of marginal medians, không median paired ratio.\n'
    bodies['canonical_tables.md']=canonical
    bodies={name:polish_prose(content) for name,content in bodies.items()}
    bodies['validation_report.md']=validation_document(tables,bodies)
    assert len(bodies)==17 and len(tables)==8
    assert all(sha(p)==h for p,h in PINS.items()),'source changed during synthesis'
    return bodies,tables

def write_narratives(main,cost):
    # No new inference/statistical fit: these narratives interpret fixed metrics.
    bodies={}
    bodies['README.md']='''# Thesis scientific synthesis

Scientific verdict: **SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS**. Experimental phase closed. Không có mandatory experiment còn lại trong central claim đã khóa.

Đọc [research story](research_story.md), rồi RQ1–RQ5: [accuracy](rq1_accuracy.md), [generalization](rq2_generalization.md), [failures](rq3_failures.md), [accuracy–cost](rq4_accuracy_cost.md), [fidelity](rq5_fidelity_applicability.md). [Canonical tables](canonical_tables.md) có tám CSV nhỏ để dùng trong thesis và presentation; không sao chép raw registries.

[Reproducibility findings](reproducibility_findings.md), [contributions](thesis_contributions.md), [limitations](limitations.md), [claims not to make](claims_not_to_make.md), [Results/Discussion guide](results_discussion_outline.md), [GĐ1 mapping](gd1_requirement_mapping.md), [18-slide storyline](presentation_storyline.md), [conclusion](thesis_ready_conclusion.md), [validation](validation_report.md).

Authority: Final RQ Closure v2. Five reporting RQs giữ đúng numbering của closure; historical Research Contract v1.1 có four RQs và được crosswalk, không sửa lại. Contract's10/21 wording là snapshot lịch sử; current grid21/21.

Tái tạo hoặc kiểm tra byte-only: `python3 -B benchmark_v1/scripts/build_thesis_synthesis.py` / thêm `--check`. Builder chỉ đọc evidence, author tables và kiểm hash. Không import model, fit, infer, preprocess circuits hay đo timings. Không commit/push, không chỉnh Canva/Notion/LaTeX. Đây là local synthesis, không approved public release.
'''+refs('closure','verdict','current')
    bodies['research_story.md']='''# Câu chuyện nghiên cứu

## Objective

Nghiên cứu so sánh các phương pháp ước lượng runtime của một circuit trên real QPU và quantum simulator, trong evaluation protocols có target semantics, input qualification và comparison population rõ ràng. Đóng góp chính là benchmark và phân tích tính áp dụng của estimator, không đề xuất một Graph model mới.

## Vì sao cần benchmark theo semantics

Một scheduled duration có thể đúng theo gate timing nhưng khác observed service time. Simulator runtime phụ thuộc engine, hardware, precision, shots, noise, approximation, API endpoint và warm/cold boundary. Một estimator tốt cho warm scalar contraction không tự dự đoán được noisy Aer execution hay CUDA-Q sample. Scheduler là motivation, chưa phải một outcome đã kiểm chứng.

## Hai tracks

Track A giữ target circuit-level observed execution/service seconds theo boundary nguồn. One-circuit job được dùng với quan hệ1:1 đã audit, không coi mọi job/batch/workflow turnaround là circuit execution. Queue time, QPU usage và end-to-end service latency không được thay thế cho label. Cohort qualified gồm340 Ma–Li,230 Qonductor,3945 QPack,165 conservative groups. QPack chiếm87,4% rows nhưng chỉ6 structural groups; reconstructed angles/layout không được gọi exact historical recovery.

Track B dùng engine/context-specific runtime labels: noisy Aer post-transpile execution, CUDA-Q Dense warm sample FP32/FP64, MPS warm state extraction và quality ladder, QCSim engine-reported component time. cuTensorNet same-plan contraction và Pasqal analog emulation là companions có endpoint riêng, không được ghép vào một leaderboard.

## Narrative theo RQ

RQ1 thiết lập accuracy trên shared held-out observations/hashes, có simple và competitive baselines. RQ2 xác định ranh giới transfer: completed negative family findings, repeated-structure backend contexts và size probes; không suy rộng mọi hardware/engine. RQ3 tổ chức failures thành prediction, representation, sampling/metric và quality-safety modes. RQ4 đo warm full pipeline prediction cost để tránh chỉ nhìn inference. RQ5 giải thích native components, adaptations và proxies, đồng thời khóa claim boundaries.

## Kết luận xuyên suốt

Graph có predictive signal trên QPU, nhưng advantage trước competitive classical pipelines nhỏ và chưa được paired uncertainty xác lập. Trong Dense exact30, MLP có accuracy và B2 tốt hơn Graph; fixed MPS lại có strong local Graph signal. Phức tạp hơn không bảo đảm phù hợp hơn. Input/context, baseline strength, support và runtime boundary quyết định ý nghĩa của phép so sánh.

Scientific sufficiency đến từ comparison có traceability và limitations rõ, không từ universal success. Không có blocking experimental gap cho các central claims này; release metadata/rights và public reproduction vẫn là gates riêng.
'''+refs('closure','claims','qpu','sim','qcec')
    bodies['rq1_accuracy.md']='''# RQ1 — Predictive accuracy and comparative effectiveness

## Question

Các estimators dự đoán runtime chính xác đến đâu so với simple, analytical và learned baselines?

## Methodology

Track A dùng chung4515 observations và five outer/four inner grouped partitions. Fits/transforms/tuning/calibration không dùng outer-test labels. Graph/MLP dùng protocol-defined per-row median-of-three, không chọn best seed. Parent neural fits chọn epoch theo inner0, fit inner1–3; regressors dùng inner selection rồi full outer refit. Vì vậy đây là pipeline comparison, không matched-budget hoặc topology-only ablation. Track B so trên đúng context và shared hashes/rungs.

## Evidence and result

Table B ghi exact metrics và nguồn. Pooled MAE của Graph0.834133328 s, MLP0.893422980 s, classical selector0.886656328 s; global train-median1.138275552 s. Graph−MLP conditional paired95% interval[−0.329074,+0.484567] s; Graph−selector[−0.317858,+0.457908] s. Point differences favor Graph nhưng cả hai intervals chứa zero. Source-balanced MAE là Graph0.898029392, MLP0.976019206, selector0.878982346 s. Equal-group MAE đổi thứ tự Graph/MLP thành0.633242057/0.580696886 s.

Analytical comparison giữ raw và outer-train-calibrated predictions riêng. QCRE/Qiskit schedule có signal nhưng không đồng nghĩa service seconds. Scholten proxy có3399 eligible observations, không phải full4515 original estimator. Calibration improvement phải so với references trên chính eligible shared rows, không với full-cohort reference.

Table D giữ context-specific results. Transpiled Aer GNN MAE0.300764122 s tốt hơn source-view0.449658482 s, nhưng XGBoost transpiled0.297035322 s là comparator cạnh tranh; không có resolved GNN superiority. Dense RF có lower local MAE hơn Graph/MLP; fixed-MPS Graph MAE0.296091793 s so Ridge1.275667370 s. Family conditioning chưa cho stable runtime advantage. Maestro CPU component, TN native EST và Pasqal companion có clock/applicability riêng.

## Interpretation

Graph có skill so với simple constant, nhưng competitive classical baseline làm lợi thế nhỏ hơn. Source/group weighting là khác estimand, không công cụ chọn bảng đẹp. Simulator input giống nhau chưa đủ: output task và measurement boundary cũng phải khớp.

## Limitation

165 groups không tương đương4515 independent structures. Bootstrap conditional theo fitted predictions, không bao gồm future-workload guarantee hoặc đầy đủ training variability. Different feature/capacity/refit procedures không cho causal topology claim.

## Thesis takeaway

RQ1 **ANSWERED WITH QUALIFICATIONS** cho cả tracks: conditional pipeline comparison có bằng chứng; universal winner không có.
'''+refs('rq','qpu','qpu_pairs','controls','sim','sim_pairs','claims')
    bodies['rq2_generalization.md']='''# RQ2 — Generalization

## Question

Estimators hoạt động thế nào ngoài training/calibration distribution?

## Methodology

Family/workload holdouts, backend-context holdouts và QPack size holdouts giữ riêng. Full grid có9 family+10 backend+2 size cases,21 CPU/21 neural cases,126 model×seed cells. Test/train observation IDs disjoint trong từng case. Backend transfer cố ý cho phép đã thấy circuit structure; family embargo không cùng nghĩa. Các cases có thể dùng lại observations:6684 test assignments phủ4515 unique IDs, không pool như6684 independent samples. Generalization runner chọn epoch internally rồi fresh full outer-train refit; degradation versus parent cũng thay training procedure.

## Evidence and result

Table C giữ42 neural method rows, support, train-median/CPU selector và paired uncertainty. QNN26 observations/14 groups và Random30/16 groups có severe overprediction: Graph/MLP MAE36.272171/50.763748 s và145.708260/188.814714 s. Train-median MAE2.658/3.627 s. Không gọi relative Graph win là useful OOD prediction.

Kyoto148/148 MLP fit-vector collisions, Osaka149/192 và Quito1051/1051 không phải observation-ID overlap. Kyoto là known-circuit context transfer; Osaka có43 new-group rows; Quito chứa1009 QPack+42 Qonductor rows trong7 groups. Graph full inputs có nominal T1/T2 nên không cùng full-input collision. Context effect vẫn confounded bởi source/workload/shots và calibration assumptions; không isolate hardware generation.

Track B có grouped-hash OOF, separate family-held-out MPS ladder và historical width study. Trained FP32/FP64 riêng không phải precision transfer. Descriptive source-balanced parent slices không phải source-held-out training experiment.

## Interpretation: ba loại evidence

1. **Evaluated failure:** QNN/Random ngoài supported feature/range distribution.
2. **Evaluated but confounded transfer:** backend-context với repeated structures hoặc source/workload changes; low-support QPack size probes.
3. **Unevaluated transfer:** controlled source/shot shift, isolated generation/topology shift và broad current simulator-engine/hardware portability.

## Limitation

Small group counts giới hạn uncertainty; không tự tạo CI khi frozen rule không cho. Per-case performance không được average R² thành global score. Historical Twolocalrandom unauthorized-at-execution disclosure vẫn giữ dù later admission.

## Thesis takeaway

RQ2 **PARTIALLY ANSWERED, NON-BLOCKING** cho hai tracks. Evaluated failures và context limitations đã trả lời phần có evidence; universal portability không cần để bảo vệ central benchmark claim.
'''+refs('rq','generalization','grid','sim','claims')
    bodies['rq3_failures.md']='''# RQ3 — Error behavior, robustness and failure modes

## Question

Estimators thất bại ở đâu, theo kiểu nào, và failures ảnh hưởng selection ra sao?

## Methodology

Giữ MAE/MedAE/RMSE/R², signed residual, tails, coverage, support và train-only controls. Không bỏ outliers, chọn seeds/metrics hoặc điều chỉnh quality threshold theo test performance. Table F liên kết failure class với evidence; Table B/C giữ denominators và distributions.

## Evidence and result

**Prediction failure:** QNN/Random overpredict mọi test observation ở cả neural pipelines, có extremely negative R². Đây là systematic error, không chỉ một outlier. Point winner tương đối không đủ practical usefulness.

**Representation failure/limitation:** Kyoto raw và post-transform MLP vectors148/148 đã có trong fit, nhưng observation IDs không trùng. Global branch không có backend/calibration context; Graph có nominal node T1/T2. Collision khác label leakage và không chứng minh learned hardware adaptation.

**Aggregation and selection sensitivity:** Source-balanced/equal-group estimands đổi ranking. Dense selected30 R² cao hơn full150; subset qualification không được biến thành full-context performance. Median-reference MedAE tốt hơn neural trên Dense timed subsets, còn MAE/RMSE/R² kém hơn. Tail errors và target variance giải thích vì sao metrics không đồng thuận, không cho phép chọn metric hậu nghiệm.

**Quality-safety failure:** Trong MPS c44, mỗi frozen family-conditioned và agnostic selector vi phạm fidelity gate trên64/140 observed choices;10/150 unassessed. Không cộng hai branches thành128 independent failures và không áp64/140 cho family-holdout variant khác. Chạy simulator xong hoặc dự đoán runtime tốt không đảm bảo state đủ fidelity.

**Context/cost failure:** TN EST/warm ratio có bias trên successful same-plan slices; Pasqal17/18 quality passes không xác nhận analytical coefficients đã identifiable. OOM, timeout hoặc unsupported input là coverage/resource/applicability outcomes, không tự là model prediction error.

## Interpretation

Negative findings là kết quả của evaluation có protocol hợp lệ. Cần phân biệt model extrapolation, missing-context collision, unavailable engine support và preprocessing incompatibility. Không quy tất cả poor scores cho GNN yếu hay implementation sai.

## Limitation

Range/support shifts chưa là causal explanation. Không có prospective uncertainty/safety guarantee hoặc complete feature intervention study.

## Thesis takeaway

RQ3 **ANSWERED WITH QUALIFICATIONS**. Report failures cùng practical usefulness; không chỉ báo ai có MAE thấp hơn.
'''+refs('rq','generalization','grid','qpu','mps','tn','pasqal','p5')
    cost_table=markdown_table(cost,['context','method','n','accuracy_unit','mae','rmse','r2','median_B2_ms'])
    bodies['rq4_accuracy_cost.md']='''# RQ4 — Accuracy–cost trade-off and scalability

## Question

Chi phí tạo một runtime prediction là bao nhiêu, và accuracy có tương xứng với chi phí đó không?

## Methodology

Full-B2 là warm online raw-input-to-prediction path: parsing, features, Graph construction khi cần, frozen transforms, serialization/IPC, tensor/H2D, inference/inverse transform/D2H và aggregation. Không bao gồm offline training/model load/cold start. So sánh genuine per-method paths ở cùng external boundary, không ép hai methods trả cùng intermediate work. Accuracy và estimator latency là hai quantities khác nhau; không lấy MAE trừ B2.

## Evidence and result

'''+cost_table+'''
QPU exact45 gồm38 Ma–Li+7 Qonductor, không QPack. Graph−MLP MAE−0.088344266 s, conditional95% interval[−0.354865074,+0.137886216] s chứa zero. Ratio of marginal B2 medians2.224255631,[2.067538851,2.408525542]. Median paired ratio2.370999 là statistic khác.

Dense FP32 exact30: Graph−MLP MAE0.230099699 ms,[0.056416868,0.459004446]; B2 ratio1.964950273,[1.916280158,2.013973472]. FP64 exact30: delta0.189680237 ms,[0.027736609,0.413599796]; ratio1.959891936,[1.897087796,2.010438803]. MLP dominates Graph trên hai criteria này, conditional vào hai measured subsets.

Preparation shares của accumulated B2: QPU Graph76.1086%,MLP97.2488%; Dense FP32 Graph67.6274%,MLP86.1642%; FP64 Graph62.7999%,MLP83.0888%. Đây là sum(stage durations)/sum(B2), không percentage của marginal median. Stage medians không được cộng hoặc trừ để suy causal overhead. Parsing/globals cũng là work thật của MLP.

## Interpretation

Prediction-only inference bỏ qua substantial online preparation. Trên QPU subset, extra Graph cost không đi cùng statistically established accuracy advantage. Dense measured pair cho lower accuracy error và lower B2 ở MLP. Không suy ra mọi Graph model vô ích.

## Limitation

Exact45≠full4515; exact30≠full150. Full150 Dense neural R² khoảng0.13–0.14, không dùng high subset R² làm full performance. Historical simulator execution cross-environment, không matched operational break-even. B1/B2 khác occasions/instrumentation, không ratio matched slowdown. Strong classical fitted-state overhead thiếu; chưa complete Pareto frontier, cold deployment, cross-hardware scaling hoặc scheduler gain.

## Thesis takeaway

RQ4 **ANSWERED WITHIN MEASURED SCOPE** ở cả tracks, mã hóa `ANSWERED WITH QUALIFICATIONS` theo closure vocabulary. Bounded accuracy–cost finding đã đủ, không yêu cầu universal scalability.
'''+refs('rq','p4','p5','dense_full','verdict')
    bodies['rq5_fidelity_applicability.md']='''# RQ5 — Method fidelity, applicability and practical selection

## Question

Method nào có thể áp dụng cho target/input/context nào, và kết quả nào không được gọi paper-exact reproduction?

## Methodology

Table A dùng finalized implementation registry. Đổi dataset không tự làm method unfaithful. Thay representation, feature semantics, transforms, objective, search/refit procedure hoặc output semantics phải khai báo. Registry classification áp ở component, không toàn paper. Scores không thay được source audit.

## Evidence and result

**Original/native estimator:** gọi native Qiskit API, pinned QCRE timing component hoặc cuTensorNet EST, giữ original output boundary. Không đồng nghĩa reproducing whole paper/workflow.

**Faithful reproduction:** phải có source/code/input/preprocessing/procedure và output compatibility đủ kiểm chứng. Không tự gán nhãn này cho current unified Graph/MLP.

**Architecture/core adaptation:** Ma–Li-style Graph/global MLP, Qonductor six-regressor pipeline, Azizov local-core GNN, Family-aware and Maestro components. Qonductor giữ upstream counter gọi `swap` nhưng đếm CX/CZ/ECR, operand-stack depth, active width, shots và circuit_count1; bounded search không nguyên full upstream search/mitigation. Source/hybrid/transpiled view của Azizov được dựng lại có qualified local inputs, không invented author encoding guarantee.

**Generalized proxy:** Scholten-inspired same-width/nominal throughput variant. Original effective depth/QV ensemble chưa verified eligible; không quy CLOPS_h thành CLOPS_v hay lấy T1/T2/gate durations giả CLOPS.

**Calibrated adaptation:** outer-train affine/log bridge từ native schedule/cost sang service label. Intercept không phải physical overhead đo trực tiếp. Hyb survival-adjusted useful-work/resource cost không tự thành one-shot execution time.

**Benchmark-defined diagnostic:** train-median references, context classical baselines và Pasqal analog companion. Diagnostic có thể hữu ích nhưng không giả làm paper-native predictor.

## Interpretation

Ma–Li raw51 gồm44 gate counters,2 size và5 structural fields. Support-based refinement cho original simulator corpus41, main local QPU40; không lấy41 vị trí đầu. Current Dense adapter dùng7 globals, không ép41 lên mọi simulator. Historical executed real-QPU tensor của tác giả chưa rõ; source-native terminology không thay evidence. Graph/MLP hiện tại là architecture/core adaptations.

Practical selection trước hết hỏi label boundary, input availability, preprocessing compatibility và quality. Native scheduled duration chưa phải observed service; TN scalar contraction chưa phải CUDA-Q sample; MPS quality policy là phần của task. Không universal winner chỉ từ MAE.

## Limitation

QCEC32 zero-input state/output agreements chỉ hỗ trợ existing recipe qualification.8 full-unitary PROVED và24 COUNTEREXAMPLE dưới assumed mappings;230 observations liên kết32 pairs, không230 independent proofs. Không exact historical recovery hay compilation history. Registry fidelity và scope limits phải đi cùng score.

## Thesis takeaway

RQ5 **ANSWERED WITH QUALIFICATIONS**. Applicability/fidelity rules đóng được trong benchmark scope, không cần production-safe universal selector để thành một benchmark có giá trị.
'''+refs('rq','registry','taxonomy','feature','qcec','claims')
    bodies['reproducibility_findings.md']='''# Reproducibility findings

## Ba mức equivalence

**Semantic circuit equivalence** xét mathematical operation hoặc declared input/output distributions. **Representation equivalence** xét graph/feature structure dưới mapping. **Frozen-model input equivalence** yêu cầu consumed tensors, order, values, masks và transforms đúng với fitted checkpoint. Hai mức đầu không tự bảo đảm mức cuối.

## Feature support

Raw schema51 không đồng nghĩa executed width51. Support refinement tạo41 cho original simulator corpus và40 cho current parent QPU. Generalization fits có width11/22/40 theo train population; một positive constant vẫn có thể tồn tại trong tensor nhưng variable mask zero. Matching feature count không chứng minh author processed tensor được phục hồi.

## Operation order

QPU raw parser instruction sequence/wires khớp45/45 giữa environments.38 Ma–Li cases diverge ở model-consumed order qua RemoveBarriers DAG-to-circuit reconstruction. Terra0.25.2 và Qiskit2.5.2 khác deterministic topological tie ordering; node_index177 làm order visible cho model.45 named dependency graphs equivalently relabeled không đủ để gọi ordered tensors compatible. RemoveFinalMeasurements giữ version-dependent order; no retrospectively selected per-row permutation.

## RCCX compatibility

Historical/library RCCX definitions U2/U1/CX và H/T/TDG/CX khác representation trước lowering. Adapter không được silently accept unvalidated basis để ép inference PASS. Declared historical dispatch/environment và frozen transforms khôi phục exact consumed arrays. Fix dựa vào independent frozen inputs, không target/prediction performance.

## Verified scope

Phase3 phục hồi210/210 independently encoded model-ready inputs:90 QPU method inputs (45×Graph/MLP) và120 Dense inputs (30×2 methods×2 precision contexts). Frozen NPZ bytes không được overwrite; transformed arrays được so exact. Numeric raw-global tolerance1e−12 chỉ cho decimal CSV storage. No model forward được dùng làm equivalence test.

## Implication

Environment-aware adapter có thể cần để replay một frozen estimator, ngay cả khi quantum circuit semantics không đổi. Original Dense producing package versions và author real tensor vẫn thiếu metadata; demonstrated compatibility không claim exact historical environment. Đây là reproducibility lesson, không causal proof topology làm accuracy tốt hơn.

## Provenance erratum

Final closure đã resolve6 README checks bằng correct package-relative namespace và182 named declarations (180 SHA256+2 Git revisions). Không chỉnh historical hashes hay raw evidence. Generic validator PASS là engineering evidence, không scientific sufficiency tự động.
'''+refs('feature','recovery','replay','provenance')
    bodies['thesis_contributions.md']=read(SOURCES['contributions']).replace('# Defensible thesis contributions','# Thesis contributions',1)+'''
## Cách dùng trong thesis

C1 mô tả framework tổ chức semantics/population/comparisons, không claim một theoretical novelty chưa chứng minh. C2 là empirical study của declared pipelines/context-specific estimators. C3 giữ chữ **warm full-B2** thay vì full production deployment. C4 nói conditional value của complexity, không graph topology vô ích. C5 nói exact-input replay và reproduction fidelity, không khẳng định recover tác giả tensor/historical logical instance.

Đóng góp chính là một benchmark có negative findings và claim boundaries kiểm chứng được, không một proposed estimator được quảng bá như universal solution.
'''+refs('contributions','claims')
    bodies['limitations.md']='''# Limitations and future work

## Scientific scope

RQ2 chỉ partially answered: complete evaluated grid không isolate source/shot, hardware generation/topology hay universal unseen circuit portability. Backends khác nhau không tự tạo controlled hardware-generation experiment. Simulator engine/hardware/precision transfer rộng chưa measured.

QPU target giữ boundary nguồn, không một pulse clock harmonized. Qualified cohort có exact Ma–Li logical inputs và reconstruction-qualified Qonductor/QPack. Historical QPack optimizer angles và Qonductor layouts chưa recovered. QCEC zero-input evidence không chứng minh full-unitary/compilation identity.165 conservative groups và QPack dominance giới hạn population interpretation.

Graph/MLP khác thông tin, capacity và historical neural/classical refit procedures. Paired uncertainty conditional theo fitted predictions/groups, không fully cover training stochasticity, multiple-comparison selection hoặc future coverage. Source-balanced/pooled estimands được report riêng.

Historical Ma–Li author real-QPU tensor provenance vẫn chưa giải; current parent40 và original simulator41 không đủ để gọi unified faithful reproduction. Scholten original effective-depth eligibility chưa verified. Native schedules, cost proxies và learned calibration bridges không interchangeable.

Track B không universal simulator leaderboard. Core q2–9/context support nhỏ, resource-qualified frontier không unbiased large-circuit sample. Maestro engine-reported time, Dense sample, MPS state, TN contraction và analog TDVP endpoints khác nhau. MPS unsafe selections và Pasqal unidentified coefficients được giữ, không xem thiếu predictive score là failed experiment cần giấu.

## Accuracy–cost scope

AM1 chỉ một recorded local hardware/software setting với warm standalone paths:45 QPU observations và30 circuits mỗi Dense precision. Không full4515 overhead, không strong classical fitted-state cost, không all-method Pareto frontier. Historical simulator execution cross-environment nên không matched break-even. Không cold startup, deployment lifecycle, scheduler throughput hay amortization validation.

## Delivery scope

Scientific sufficiency không release approval. Rights/owner/citation metadata, private-data policy, controlled export và fresh clean-checkout verification phải giải riêng. Public reduced QPU payload không self-contained raw-data refit package. Synthesis này không thực hiện hay nhận lại PASS của các release gates.

## Future work

Table H giữ9 gaps đã khóa. Tất cả NON-BLOCKING cho central scientific claims; public release có independent delivery gates. Broader transfer, competitive cost frontier, safe selection và causal interventions là possible future scope, không nhiệm vụ tiếp tục old queue. Không có required additional experiment.
'''+refs('gaps','rq','registry','qcec','verdict')
    bodies['claims_not_to_make.md']='''# Claims not to make

| Không được claim | Formulation có evidence |
|---|---|
| Graph is the best quantum runtime estimator | Pooled point MAE tốt hơn một số comparators trên qualified4515; competitive paired advantage unresolved |
| Graph topology is proven to improve prediction | Complete Graph/global pipelines khác input/capacity; không topology-only causal proof |
| Every original paper was exactly reproduced | Native components, reimplementations/adaptations, proxies và companions được registry phân biệt |
| Estimator generalizes to unseen hardware | Evaluated backend contexts có repeated structures/confounding; isolated generation effect chưa tested |
| Scholten was faithfully reproduced | Current scored pipeline là Scholten-inspired generalized proxy; original eligibility chưa verified |
| Scheduled duration equals observed execution time | Native schedule khác source-retained execution/service labels; outer-train bridge là estimator riêng |
| MLP universally dominates Graph | MLP dominates Graph trên MAE/warm B2 trong hai Dense exact30 subsets |
| Estimator overhead operationally exceeds simulator runtime | Cross-environment simulator timing không matched operational break-even |
| Benchmark proves scheduler throughput gain | Scheduler là motivation; chưa scheduler-level validation |
| All simulators share one winner | Engine/API/precision/quality/clock-specific comparisons |
| High Dense subset R² is full-context performance | Exact30 và full150 metrics tách riêng; full150 neural R² khoảng0.13–0.14 |
| 21 cases mean universal generalization | Completed tested grid, không universal source/hardware/engine portability |
| 6684 test assignments are independent observations | Cases overlap;4515 unique IDs, within-case partitions disjoint |
| QCEC recovered historical logical circuits | Zero-input state/output checks under declared mappings;8 unitary proofs/24 counterexamples |
| Kyoto collision is observation leakage | Feature equality không observation-ID overlap; context-blind representation collision |
| Affine intercept measures physical launch overhead | Statistical outer-train bridge, không isolated physical component experiment |
| Ratios or differences of stage medians identify causal cost | Accumulated stage fractions/correlations descriptive; distinct marginal/paired statistics |
| Validation PASS means publish/retrain reproducibility approved | Saved-evidence integrity, scientific sufficiency và release rights/reproduction là separate gates |

Không diễn giải ZX inconclusive thành non-equivalence hoặc24 full-unitary counterexamples thành prepared-state mismatch. Không gọi230 linked observations là230 independent proofs.
'''+refs('claims','registry','verdict','qcec')
    bodies['results_discussion_outline.md']='''# Results and Discussion writing guide

## Results

1. **Study populations and protocol:** Track A4515/source qualification/groups/outer-inner splits; Track B per-context task/API/quality/resource boundaries. Declare model features, refit/search budgets, seeds and ensemble semantics before scores.
2. **RQ1 accuracy:** Tables A/B/D, same-row comparator results, raw/calibrated analytical coverage và conditional paired intervals. Source-balanced views alongside pooled. No universal winner headline.
3. **RQ2 transfer:** Table C per-family/backend/size cases, train-only references, CPU selector, structural support, collisions và per-seed dispersion.21-case coverage không pooled OOD score.
4. **RQ3 failures:** Table F distributions/tails, QNN/Random negative R², collisions, unsafe MPS counts, subset/full-context divergence, unavailable resource outcomes. Preserve all denominators.
5. **RQ4 cost:** Table E exact45/exact30, MAE/RMSE/R² units versus estimator B2 milliseconds, conditional intervals và accumulated preparation fractions. No utility subtraction.
6. **RQ5/replay:** Table A classification và210/210 exact-input recovery; report what was verified, not inferred historical history.

Results trình bày observed records, conditions, metrics, coverage và uncertainty. Nếu explanation chưa experimentally isolated, chuyển sang Discussion và gắn descriptive/hypothesis label.

## Discussion

1. **Baseline strength:** Graph vs constant khác Graph vs selected strong classical pipeline; information/capacity/refit asymmetry.
2. **Support and failure:** Known-circuit transfer/collisions, OOD range shifts và source dominance giới hạn generalization.
3. **Systems cost:** Warm preparation là work có thật; distinguish cached inference, fullB2 và unmeasured deployment lifecycle.
4. **Fidelity/prior work:** Đối chiếu audited paper components/targets/inputs trong registry, không compare local R² với paper score như same environment. Explain adaptations cụ thể, không quy mọi discrepancy cho dataset mới.
5. **Cross-track practical selection:** target semantics, input support, quality, baseline và cost là decision criteria; chưa universal automated selector.
6. **Contributions/limits:** C1–C5 qualified formulations, RQ closure và independent release gates. No new experiment plan is implied.

## Conclusions

Trả lời từng final reporting RQ, dùng một bounded takeaway. Kết luận bằng scientific sufficiency với material limitations và thesis contributions, không chronological task completion diary.
'''+refs('rq','claims','registry','contributions')
    bodies['gd1_requirement_mapping.md']='''# Project and GĐ1 requirement mapping

## Requirement provenance

Trong repo inspected không tìm thấy official GĐ1 rubric/approved institutional requirement document có thể xác minh. Không ghi những hàng dưới là quoted official GĐ1 requirements và không tự chấm pass rubric. Đây là documented-project mapping dựa trên Research Contract v1.1, current conclusions, Final RQ Closure v2 và existing120-question evidence crosswalk. Institutional wording/formatting approval remains **NOT VERIFIED**, không blocking experiment.

Historical contract có4 RQs: pipeline comparison, analytical timing, representation/context và generalization. Final synthesis dùng5 reporting RQs đã approved; historical RQ4→final RQ2, AM1→final RQ4. No contract rewrite. Current21/21 completion supersedes old10/21 narrative only, not historical execution authority.

| Documented project requirement | Evidence | Thesis section | Completion status |
|---|---|---|---|
| Benchmark runtime predictors for real QPU | qualified4515, shared OOF, Table B, analytical eligible controls | Methodology; Results RQ1A/RQ5A | COMPLETE WITH QUALIFICATIONS |
| Benchmark simulator runtime estimators | per-context saved labels/OOF, Table D; native/analog companions | Methodology; Results RQ1B/RQ5B | COMPLETE WITH QUALIFICATIONS; no universal engine claim |
| Explain comparable inputs/targets/evaluation | frozen contracts, input qualification, registry, Tables A/G | Dataset and Methodology | COMPLETE WITH DECLARED SOURCE/INPUT LIMITS |
| Analyze robustness/generalization | complete21 CPU/neural cases, Table C/F and support | RQ2/RQ3 | PARTIALLY ANSWERED, NON-BLOCKING broad portability |
| Relate estimation accuracy to prediction cost | AM1 matched45/30 and Table E | RQ4 | ANSWERED WITHIN MEASURED SCOPE |
| Reproducibility and experiment traceability | exhaustive saved-evidence reconciliation; closure provenance;210 exact inputs | Reproducibility appendix; RQ5 | SAVED-EVIDENCE VERIFIED; release/raw-refit separate |
| Defensible contributions and limitations | C1–C5, Tables G/H, forbidden claims | Discussion and Conclusion | COMPLETE FOR SCIENTIFIC SYNTHESIS |
| Official GĐ1 administrative requirements | no verified official rubric in consulted repo sources | Front matter/administrative checklist | NOT VERIFIED; no invented requirement |

Existing crosswalk is research-evidence classification, not institutional grading rubric. Table hashes and source links permit supervisor review without pretending unavailable official authority.
'''+refs('contract','current','rq','crosswalk','contributions')
    slides=[
        ('Problem','Cần runtime estimates cho planning; scheduler benefit là motivation, chưa measured outcome.','research_story.md','closure'),
        ('Runtime semantics','Observed QPU service, schedule, simulator sample/state/contraction là clocks khác nhau.','rq5_fidelity_applicability.md','registry'),
        ('Research gap','Không có fair universal ranking khi clock/input/quality và source support chưa xác định.','research_story.md','claims'),
        ('Benchmark design','Same held-out rows per method/context; grouped partitions; train-only transforms/tuning/calibration.','canonical_tables.md','contract'),
        ('Five RQs','Accuracy, generalization, failures, accuracy–cost, fidelity; preserve closure numbering.','canonical_tables.md','rq'),
        ('Datasets and contexts','4515=340+230+3945,165 groups; simulator contexts riêng và qualified survival support.','research_story.md','current'),
        ('Estimator taxonomy','Native component vs architecture/core adaptation vs proxy/calibrated bridge/diagnostic.','canonical_tables.md','taxonomy'),
        ('Track A accuracy','Graph pooled MAE0.8341s; MLP0.8934; selector0.8867. Competitive paired CIs cross zero.','rq1_accuracy.md','qpu'),
        ('Source/group composition','Source-balanced selector point better; equal-group MLP/Graph ordering reverses.','rq1_accuracy.md','qpu'),
        ('Generalization and failures','21-case completion; QNN/Random catastrophic errors; Kyoto collision ≠ ID leakage.','rq2_generalization.md','generalization'),
        ('Track B context results','Aer compiled information, Dense strong classical comparator, fixed MPS signal; no cross-engine winner.','canonical_tables.md','sim'),
        ('Quality safety','Each MPS c44 selector64/140 unsafe choices;10 unassessed; runtime success insufficient.','rq3_failures.md','mps'),
        ('Why full-B2','Raw parse/features/graph/IPC/inference stages; accumulated preparation fractions.','rq4_accuracy_cost.md','p4'),
        ('QPU accuracy–cost','Exact45: accuracy advantage unresolved; warm B2 Graph≈2.22× MLP.','canonical_tables.md','p4'),
        ('Dense accuracy–cost','Exact30 each precision: MLP lower MAE/B2, conditional paired CIs; full150 neural R²≈0.14.','canonical_tables.md','p5'),
        ('Replay and fidelity','51 raw support-refined; operation order/RCCX environment semantics;210/210 exact inputs. QCEC no historical recovery.','reproducibility_findings.md','replay'),
        ('Contributions and limits','Five bounded contributions; portability/cold/classical cost/frontier and release gates remain distinct.','thesis_contributions.md','contributions'),
        ('Conclusion','Semantics/support/baselines/cost determine estimator applicability. Scientifically sufficient with material limitations.','thesis_ready_conclusion.md','verdict'),
    ]
    storyline='# Presentation storyline — 18 slides\n\nNội dung để người làm slide sử dụng; không tạo/ghi Canva. Chỉ copy numbers từ canonical CSV, giữ units và denominator cùng slide.\n'
    for i,(title,message,local,key) in enumerate(slides,1):
        storyline+=f'\n## {i}. {title}\n\nThông điệp: {message}\n\nNội dung: [{local}]({local}). Evidence: [{key}](../../../{SOURCES[key]}), SHA-256 `{sha(pin(SOURCES[key]))}`.\n'
    storyline+='''
## Những conclusions cũ phải thay

Không còn current10/21 hoặc12/21 generalization, cũng không primary8767/7350 cohort. Dùng4515 và21/21 grid với limits. Không dùng R² của AM1 selected30 cho full150. B1-only latency chưa full estimation cost. Không gọi41 là width của mọi simulator adaptation, originalScholten reproduced, Graph universal winner hay exact historical logical recovery. Old deck historical findings không bị sửa ở bước này.
'''
    bodies['presentation_storyline.md']=storyline
    bodies['thesis_ready_conclusion.md']='''# Thesis-ready conclusion

Nghiên cứu xây dựng và phân tích một benchmark cho quantum runtime estimators trên hai tracks: circuit-level observed execution/service time của real QPU và runtime của simulator dưới execution context xác định. Đóng góp nằm ở comparison protocol có runtime semantics, input qualification, shared-row evaluation, baseline strength, quality constraints và evidence traceability rõ ràng.

Trên4515 qualified QPU observations, Ma–Li-style Graph có pooled MAE thấp hơn global MLP và classical selector trong point estimate. Paired uncertainty không xác lập ưu thế trước hai comparators cạnh tranh, và source/group weighting thay đổi ranking. Full21-case generalization grid ghi nhận severe QNN/Random overprediction và context-blind MLP collisions trên một số backend holdouts. Những negative findings xác định ranh giới của các pipelines đã đo, không chứng minh universal unseen-hardware transfer.

Simulator results phụ thuộc target/API, noise, precision, engine và approximation. Compiled Aer information, Dense classical controls và fixed-MPS Graph có predictive signal khác nhau. Frozen MPS quality selections có failures đáng kể. Native TN estimates và analog MPS companions giữ scope riêng. Không có một simulator winner được suy ra từ các heterogeneous contexts.

AM1 warm full-B2 cho thấy parsing/preparation/bridge là substantial cost của việc tạo prediction. Trên45 QPU observations, Graph có lower point error nhưng accuracy interval chứa zero với B2 khoảng2.22× MLP. Trong30-circuit Dense FP32 và FP64 subsets, MLP có lower MAE và lower B2 với conditional paired evidence. Đây là bounded accuracy–cost comparison, không operational break-even hoặc all-method deployment frontier.

Reproduction fidelity cần phân biệt semantic circuit equivalence, representation equivalence và exact frozen-model inputs. Support refinement, graph ordering và RCCX library compatibility có thể thay consumed tensors dù circuit semantics tương thích.210/210 Phase3 inputs được phục hồi theo frozen evidence; điều này không chứng minh exact historical logical recovery hoặc original author real tensor provenance.

Final RQ1/RQ3/RQ5 được answered with qualifications, RQ4 answered within measured scope và RQ2 partially answered nhưng non-blocking. Verdict là **SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS**. No mandatory additional experiment remains. Experimental phase closed; thesis synthesis, scoped presentation và independent rights/release validation là bước bàn giao, không tiếp tục experiment queue.
'''+refs('rq','claims','verdict','p4','p5','replay')
    return bodies

def validation_document(tables,bodies):
    checks={"status":"PASS","csv_tables":8,"narrative_files":17,"rq_track_rows":10,
        "table_rows":{name:len(data) for name,data in tables.items()},
        "markdown_local_links_checked":sum(len(re.findall(r'\[[^\]]+\]\(([^)]+)\)',content)) for content in bodies.values()),
        "closure_status_exact":True,"blocking_experimental_gaps":0,
        "model_ready_recovery_evidence":"210/210, read saved receipt only",
        "six_readme_and_182_hash_resolution":"preserved closure erratum; not rerun workloads",
        "source_files_pinned":len(PINS),"source_hashes_unchanged":True,
        "new_metrics_or_bootstrap":False,"new_model_execution":False,"new_timing_or_quantum_execution":False,
        "protocol_prediction_checkpoint_or_historical_artifact_changes":False,
        "commit_push_canva_notion_latex":False,"official_GD1_rubric":"unavailable in consulted sources; not invented"}
    return '# Synthesis validation\n\n```json\n'+json.dumps(checks,indent=2,ensure_ascii=False)+'\n```\n\n'+'''
## Fresh checks for this package

Builder checks current closure-pinned source hashes, copied status identity, exact table values/units, QPU 4515/full Dense 150 versus AM1 exact45/exact30, saved paired interval directions, MPS 64/140 scoped to c44, Pasqal 17/18 quality cells and 210/210 saved input recovery. No inference, preprocessing replay, simulation or statistical re-fitting occurs. CSV authoring uses bundled Artifact Tool typed ranges; values are serialized as plain CSV, not formatted numbers.

`--check` regenerates narrative/table bytes in memory and checks the saved package, source hashes and local Markdown links. CSV source path/hash columns are included for every row. Table C retains21×2 neural cases; supporting CPU selectors/train-only references remain scoped by case. Source-balanced/equal-group quantities are not renamed pooled accuracy. Stage fractions are accumulated sums, not medians. Cold/production overhead is not claimed.

Final closure's prior4 validators and exhaustive1268 passed/1 skipped are historical validation receipts, not new suite executions here. Fresh validation concerns this synthesis, not public release or independent raw-to-refit reproduction. Six earlier README discrepancies are namespace false positives resolved by closure; historical exhaustive report remains unchanged.

After build/check, command exits only if file inventory, CSV replay, local links, all captured source hashes and `git diff --check` pass. It prints fresh outcome/counts. No export to public tree or Git mutation.

## Source identities

'''+markdown_table([dict(path=str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else p,sha256=h) for p,h in sorted(PINS.items())],['path','sha256'])

def validate_payload(bodies,tables):
    expected_md=set(bodies)
    assert {p.name for p in OUT.glob('*.md')}==expected_md
    assert {p.name for p in (OUT/'canonical_tables').iterdir()}==set(tables)
    links=0
    for name,content in bodies.items():
        assert (OUT/name).read_text()==content,f'narrative byte drift: {name}'
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',content):
            if target.startswith(('https://','http://')):continue
            assert (OUT/target).resolve().exists(),(name,target)
            links+=1
    for name,data in tables.items():
        actual=list(csv.DictReader(io.StringIO((OUT/'canonical_tables'/name).read_text())))
        assert len(actual)==len(data),(name,len(actual),len(data))
        for a,e in zip(actual,data):
            assert list(a)==list(e),(name,'column order')
            for key,value in e.items():
                if isinstance(value,(int,float)):
                    assert float(a[key])==float(value),(name,key,a[key],value)
                else: assert a[key]==str(value),(name,key)
    assert all(sha(p)==h for p,h in PINS.items()),'source preservation failure'
    result=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return {'status':'PASS','markdown_files':len(bodies),'CSV_tables':len(tables),'table_rows':{k:len(v) for k,v in tables.items()},
        'local_links_checked':links,'sources_unchanged':True,'git_diff_check':'PASS','frozen_source_count':len(PINS)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    bodies,tables=build()
    # All reads complete before any writes; historical evidence never an output target.
    if not args.check:
        OUT.mkdir(parents=True,exist_ok=True)
        for name,content in bodies.items(): (OUT/name).write_text(content,encoding='utf-8')
        payload={"output_dir":str(OUT/'canonical_tables'),"tables":tables}
        result=subprocess.run([NODE,str(ROOT/'benchmark_v1/scripts/export_thesis_tables.mjs')],
            input=json.dumps(payload,ensure_ascii=False),text=True,capture_output=True,cwd=ROOT,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
    print(json.dumps(validate_payload(bodies,tables),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
