"""Render/replay a research-closure assessment from frozen evidence only.

No model imports, fits, inference, simulation, checker execution or file writes.
--emit-patch prints a patch for new derived reports; apply it explicitly.
Scientific status/action judgements are declared below, not inferred from PASS.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/research_question_closure/v1"
T = "artifacts/thesis_interpretation/v1/"
D = "artifacts/benchmark_deep_dive/"
P = "artifacts/real_qpu/qualified_logical_panel/"
G = "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/"

SOURCES = {
    "contract": "docs/research_contract.md",
    "freeze": "artifacts/final_research/rq4_report.md",
    "conclusions": "docs/research_conclusions.md",
    "thesis": T + "final_scientific_interpretation.md",
    "claims": T + "evidence_to_claims.csv",
    "registry": T + "estimator_fidelity_registry.md",
    "thesis_index": T + "README.md",
    "thesis_validation": T + "validation_report.md",
    "thesis_receipt": T + "validation_receipt.json",
    "thesis_manifest": T + "provenance_manifest.json",
    "addendum": "artifacts/benchmark_interpretation/v1/README.md",
    "question_matrix": "artifacts/benchmark_interpretation/v1/benchmark_question_matrix.csv",
    "feature_support": "artifacts/benchmark_interpretation/v1/generalization_field_support.csv",
    "transforms": "artifacts/benchmark_interpretation/v1/effective_input_widths.csv",
    "qpu": D + "tables/qpu_metrics_v1.csv",
    "qpu_pairs": D + "tables/qpu_shared_reference_comparisons_v1.csv",
    "qpu_shared": P + "summary/shared_row_comparisons.csv",
    "controls": D + "tables/analytical_controls_v1.csv",
    "profile": P + "summary/dataset_profile.json",
    "parent_protocol": "benchmark_v1/protocol/qualified_logical_benchmark.json",
    "parent_runner": "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
    "generalization_protocol": "benchmark_v1/protocol/real_qpu_generalization.json",
    "completion_protocol": "benchmark_v1/protocol/rq4_completion.json",
    "generalization": G + "method_metrics.csv",
    "generalization_pairs": G + "reader_tables/paired_comparisons.csv",
    "generalization_support": G + "reader_tables/input_support.csv",
    "generalization_sources": G + "reader_tables/source_metrics.csv",
    "generalization_lineage": G + "reader_tables/cell_lineage.csv",
    "generalization_manifest": G + "manifest.json",
    "sim": D + "tables/simulator_metrics_v1.csv",
    "sim_pairs": D + "tables/simulator_paired_v1.csv",
    "sim_slices": D + "tables/simulator_slices_v1.csv",
    "sim_widths": D + "tables/simulator_target_width_distributions_v1.csv",
    "sim_producing": D + "scripts/simulator.py",
    "sim_protocol": "benchmark_v1/protocol/simulator_predictor_cells_v2.json",
    "joint_protocol": "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json",
    "quality": D + "tables/mps_selection_safety_v1.csv",
    "tn": D + "tables/tn_same_plan_slices_v1.csv",
    "tn_scaling": D + "diagnostics/tn_scaling_v1.json",
    "tn_telemetry": "work/repo_finalization/publication/data/simulator/tensor_network/measurement/cutensornet_scalar_common_raw.csv",
    "analog": D + "tables/pasqal_stage_quality_v1.csv",
    "qcec": "artifacts/real_qpu/provenance_equivalence/results.json",
    "launch": "artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/REPORT.md",
    "width_study": "experiments/simulator_runtime_v2/README.md",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def link(key):
    return "[" + key + "](" + str(ROOT / SOURCES[key]) + ")"


def refs(*keys):
    return "; ".join(link(k) for k in keys)


def load_sources():
    pins = {str(ROOT / p): sha(ROOT / p) for p in SOURCES.values()}
    for p in sorted((ROOT / (P + "neural")).glob("fold_*/*/seed_*/manifest.json")):
        pins[str(p)] = sha(p)
    assert len([p for p in pins if "/neural/fold_" in p]) == 30
    return pins


def rows(key):
    with (ROOT / SOURCES[key]).open(newline="") as f:
        return list(csv.DictReader(f))


def obj(key):
    return json.loads((ROOT / SOURCES[key]).read_text())


def facts():
    claims = {r["claim_id"]: json.loads(r["exact_metrics_json"]) for r in rows("claims")}
    profile = obj("profile")
    assert profile["rows"] == 4515 and profile["groups"] == 165
    assert profile["source_counts"] == {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 230, "qpack_mcp": 3945}
    assert len(rows("qpu")) == 34 and len(rows("sim")) == 49
    assert len(set(r["context_id"] for r in rows("sim"))) == 7
    assert math.isclose(claims["C01"]["skill"], 1-claims["C01"]["graph_mae"]/claims["C01"]["median_mae"], abs_tol=1e-12)
    assert math.isclose(claims["C02"]["skill"], 1-claims["C01"]["graph_mae"]/0.8866563281384384, abs_tol=1e-12)
    tn = {r["slice"]: float(r["ratio_median"]) for r in rows("tn") if r["slice_type"] == "stratum"}
    assert math.isclose(tn["core_q2_q9"], 3.7340122316333324, rel_tol=1e-12)
    assert math.isclose(tn["frontier_q10_q16"], 4.132415604942254, rel_tol=1e-12)
    costs = {}
    for method in ("mali_logical_graph", "mali_global_mlp"):
        records = [json.loads(p.read_text()) for p in sorted((ROOT / (P + "neural")).glob("fold_*/"+method+"/seed_*/manifest.json"))]
        assert len(records) == 15 and all(r["status"] == "complete" for r in records)
        assert all(r["identity"]["gpu"] == "NVIDIA GeForce RTX 5070 Ti" for r in records)
        seconds = [r["elapsed_seconds"] for r in records]
        costs[method] = dict(cells=15, min_seconds=min(seconds), median_seconds=statistics.median(seconds), max_seconds=max(seconds), sum_cell_seconds=sum(seconds))
    # Bounds are inspected statically. The elapsed field is frozen before test prediction.
    code = (ROOT / SOURCES["parent_runner"]).read_text().split('def neural_cell(',1)[1].split('def neural_run(',1)[0]
    assert code.index('elapsed = elapsed_before + time.monotonic()-begin') < code.index('for identity in parts["test"]:')
    assert 'elapsed_seconds=elapsed,fit_rows=' in code
    telemetry = rows('tn_telemetry')
    first = [r for r in telemetry if r['observation_kind']=='first' and r['status']=='ok']
    assert len(first)==603 and len({(r['panel_member_id'],r['session_id']) for r in first})==603
    warm_by_session = {}
    for r in telemetry:
        if r['observation_kind']=='warm' and r['status']=='ok':
            warm_by_session.setdefault((r['panel_member_id'],r['session_id']),[]).append(float(r['t_warm_contract_s']))
    assert len(warm_by_session)==603 and all(len(v)==5 for v in warm_by_session.values())
    phases={k:dict(n=len(first),median_seconds=statistics.median(float(r[k]) for r in first),
                   min_seconds=min(float(r[k]) for r in first),max_seconds=max(float(r[k]) for r in first))
            for k in ('t_network_build_s','t_path_search_s','t_first_contract_s')}
    warm=[statistics.median(v) for v in warm_by_session.values()]
    phases['t_warm_contract_s']=dict(n=len(warm),median_seconds=statistics.median(warm),min_seconds=min(warm),max_seconds=max(warm))
    phases['end_to_end_nonempty_rows']=sum(bool(r['t_end_to_end_s']) for r in telemetry)
    assert phases['end_to_end_nonempty_rows']==0
    return dict(claims=claims, profile=profile, training_receipts=costs, tn_ratios=tn, tn_phases=phases)


# Ten independent Track x RQ decisions. These assess the broad working question;
# context-level decisions below may be narrower or unanswered.
CARDS = [
dict(rq="RQ1", track="A", question="Existing estimators dự đoán archived circuit service time chính xác đến đâu?", why="Kiểm tra predictive benefit, không lấy độ phức tạp model làm giá trị nghiên cứu.", minimum="Shared held-out IDs, train-only preprocessing/tuning/calibration, strong simple/classical controls, errors/coverage và paired uncertainty.", population="4.515 observations = Ma–Li 340 + Qonductor 230 + QPack 3.945; 165 conservative groups.", target="Archived observed execution/service seconds; source boundaries retained, không pulse duration chung.", split="5 outer / 4 inner grouped folds; seed 42. Neural seeds 42/1234/31415, frozen per-row median; regression six families/307 candidates.", estimators="Ma–Li-style Graph/global MLP; Qonductor-style six regressors/selector; QCRE, Qiskit, Hyb, Scholten proxy và raw/calibrated variants; train-only references.", evidence="34 metric rows; matched analytical controls; saved paired intervals và source/group-balanced metrics.", finding="Graph MAE 0,83413 s, MLP 0,89342 s, selector 0,88666 s. Graph skill 26,7% vs global median nhưng 5,9% vs selector; selector/MLP CIs chứa 0. Equal-source ranking đổi.", threats="Reconstructed tiers, heterogeneous clocks, 87,4% QPack rows/sáu structures; unequal inputs/capacity/tuning/refit procedures. Đây là pipeline benchmark, không equal-information architecture contest.", missing="Original Scholten eligibility; exact-paper reproductions; universal model ranking. Không thiếu primary shared-row scores.", impact="Không chặn bounded pipeline comparison; chặn claim mọi original method đã evaluated hoặc Graph universally best.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["qpu","qpu_pairs","qpu_shared","controls","profile","parent_protocol","registry"], gaps="G03;G05;G08"),
dict(rq="RQ1", track="B", question="Estimator nào có accuracy hữu ích trong từng simulator task/context?", why="Engine, API và approximation thay đổi target; cùng đơn vị seconds không bảo đảm comparability.", minimum="Same exact hashes/folds/engine/precision/shots/clock; references và paired comparisons; quality/terminal denominator.", population="150 exact core hashes q2–9; Maestro 143/150; joint ladder 840/900 measured rungs thuộc 140/150 hashes; TN 603/612 paired plan sessions; analog 18 cells.", target="Warm sample, warm get_state, after-transpile Aer, QCSim engine-reported time, scalar contraction và analog run tách riêng.", split="Context-specific grouped five-fold OOF; three-seed medians where frozen; separate family-component holdout for ladder; native TN same-plan pairing.", estimators="Graph/global models; Azizov source/hybrid/transpiled; family-conditioned/agnostic; Maestro-style component; native RUNTIME_EST; classical/complexity controls; Pasqal diagnostic only.", evidence="49 metrics/seven predictive contexts, paired tables, TN ratios và analog quality/time; không cross-context leaderboard.", finding="Dense RF thấp MAE hơn Graph; fixed-MPS Graph tốt hơn Ridge; Aer compiled view giúp nhưng GNN superiority over XGBoost chưa resolve; Maestro local component có skill; family conditioning chưa resolve gain.", threats="Small-width/resource-qualified survivor panel; incomplete ladders; different information/budgets; QCSim engine clock không full wall-clock. Pasqal chưa có predictive validation.", missing="Full Maestro selector/closed engine, original Quantum Rings reproduction, Pasqal predictive coefficients; broad common wall-clock engine comparison.", impact="Không chặn evaluated matched-context findings; chặn original-paper/full-wall-clock/global engine winner claims.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["sim","sim_pairs","quality","tn","analog","sim_producing","registry"], gaps="G03;G05;G06"),
dict(rq="RQ2", track="A", question="Generalize đến circuits, families, sizes, backends hoặc sources chưa thấy đến đâu?", why="Random/grouped interpolation không thay thế domain transfer.", minimum="Leakage-safe groups và predeclared OOD splits; support/composition, repeated-input checks, train-only references, paired uncertainty.", population="Parent 165 groups; full grid chín family/mười backend-context/hai size cases; 6.684 assignments nhưng 4.515 unique IDs.", target="Source-retained service seconds trong từng case, không pool overlapping cases.", split="21 frozen cases; family structural embargo; intentional structural overlap for backend transfer; QPack sizes có 3/2 test groups; 126 Graph/MLP×seed cells.", estimators="Graph/MLP median-of-three; CPU selectors/analytical routes và train-only references, case-specific fresh fits.", evidence="21/21 CPU + neural cases; test-ID hashes, lineage, shared-row and known/new support tables.", finding="QNN/Random neural transfer thất bại nghiêm trọng; CPU selector vẫn hữu ích. Kyoto là known-circuit backend context; Osaka mixed known/new; Quito mostly repeated QPack structures.", threats="Source/family/clock/shots/generation confounding; low backend/size groups; no source-held-out study; nominal-index calibration; overlapping case assignments.", missing="Systematic source/shot transfer và isolated hardware-generation effect; representative large-size evaluation.", impact="Không phủ nhận completed negative OOD findings; không đóng toàn bộ generalization agenda chỉ bằng 21/21 completion.", status="PARTIALLY ANSWERED", action="DEFER WITH EXPLICIT LIMITATION", sources=["freeze","generalization","generalization_pairs","generalization_support","generalization_sources","generalization_protocol","profile"], gaps="G04;G06;G07"),
dict(rq="RQ2", track="B", question="Generalize qua circuits, families, width và simulator/hardware contexts đến đâu?", why="Hardware-calibrated and learned predictors có thể chỉ nội suy trong calibration support.", minimum="Hash-grouped splits; dedicated OOD held-outs; train-only calibration and matched clocks/tasks; structural support.", population="Core q2–9 hashes; family-component MPS ladder; historical PyTorch dense width corpus và CUDA-Q RX-only precision/shot pilot, không nhập làm common-panel scores.", target="Separate engine/task clocks; historical kernel target khác current sample/get_state.", split="Primary hash OOF; separate ladder family holdout; historical width holdouts. Precision-specific training không phải precision transfer.", estimators="Per-context learned/empirical models, family conditioning; historical work-scaled models; native TN diagnostics.", evidence="Ladder family-holdout scores, width slices, historical width-held-out errors, same-plan TN core/frontier.", finding="Family-conditioning advantage chưa resolve. Historical dense width holdout có boundary failures; không thể chuyển score đó sang current CUDA-Q/Aer/engine khác.", threats="Một local resource profile, little width support; context-specific survivor sets; no common trained cross-engine transfer; descriptive width slices không OOD experiment.", missing="Current-context systematic width/engine/hardware transfer và unseen hardware calibration efficacy.", impact="Core local accuracy còn hợp lệ; broad portable simulator predictor/general scaling chưa answered.", status="PARTIALLY ANSWERED", action="REANALYZE SAVED EVIDENCE", sources=["sim","sim_slices","sim_widths","joint_protocol","width_study","launch","tn_scaling"], gaps="G02;G06;G07"),
dict(rq="RQ3", track="A", question="Errors tập trung ở đâu, khi nào estimators unreliable?", why="MAE trung bình có thể che catastrophic failures và deployment limits.", minimum="Signed errors, quantiles/max, absolute/relative errors, per-source/case support, simple references, collision/preprocessing audits.", population="Parent 4.515 rows và từng independent declared holdout; QNN 26 rows/14 groups, Random 30/16, Kyoto 148/117.", target="Residual = prediction minus observed service; route-specific finite coverage.", split="Saved parent OOF + frozen OOD cases; no seed/row removal after errors.", estimators="All parent routes; Graph/MLP/CPU within each case.", evidence="Tails, MedAE/RMSE/R², overprediction counts; QNN/Random distributions; raw/post-transform collisions; Hyb missingness.", finding="Cả neural models overpredict mọi QNN/Random row; R² âm nghiêm trọng. Kyoto 148/148 MLP collisions không ID leakage. Source weighting changes ranking; analytical missing values không zero-filled.", threats="Support-shift association không causal feature mechanism; fitted-model conditional uncertainty; model failure khác representation limitation.", missing="Prospective intervals, causal per-feature failure attribution, scheduler loss; không thiếu evidence rằng observed failures xảy ra.", impact="Negative findings close bounded failure questions; no prospective safety or isolated mechanism claim.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["qpu","freeze","feature_support","generalization_support","controls","thesis"], gaps="G09;G08"),
dict(rq="RQ3", track="B", question="Simulator estimators sai hoặc không an toàn dưới điều kiện nào?", why="Runtime accurate vẫn có thể chọn approximation không đạt fidelity.", minimum="Signed/tail errors per-context; terminal/quality coverage; same-rung target; support/slicing/first-warm boundaries.", population="Core contexts, joint ladder 140 assessed/150 assigned hashes, TN plan sessions, analog grid; historical frontier outcomes riêng.", target="Each context residual; quality violations separate from seconds errors; EST/warm dimensionless ratio.", split="Saved OOF and family-holdout predictions; quality measured outside timing; repeated rungs cluster by hash.", estimators="Runtime predictors, conditioned/agnostic selectors, native EST và diagnostic controls.", evidence="Context error slices, 64/140 selected quality failures, ten unassessed, 138/140 achievable hashes; TN bias and resource records.", finding="Runtime performance không bảo đảm quality selection. TN core median EST/warm 3,734, frontier 4,132; under locked plan/precision, không whole-API time estimate.", threats="Quality-incomplete hashes and survivor support; small analog grid; internal TN model opaque; no physical overhead decomposition.", missing="Prospective quality-safety guarantees, broad hardware regime and exact failure causality.", impact="Observed negative safety/bias findings đã answered; missing guarantees limit deployment claims.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["sim","sim_slices","quality","tn","tn_scaling","analog","joint_protocol"], gaps="G06;G09"),
dict(rq="RQ4", track="A", question="Accuracy có đáng đổi lấy feature/transpile/train/calibration/inference cost không?", why="Scheduler phải trả chi phí estimator; target circuit runtime không phải overhead của predictor.", minimum="Matched-query end-to-end and stage timings, cold/warm, CPU/GPU/RAM, serialized models/calibrators, prediction batch; accuracy trên cùng rows; amortization assumptions.", population="Frozen parent 30 neural cells và datasets/compiled views; 21-case receipts là execution records, không matched latency benchmark.", target="Estimator computation seconds, peak memory and preparation/training costs; separate from service labels.", split="Historical elapsed fields + producing-code timer inspection; không có dedicated matched cost evaluation split.", estimators="Graph/MLP; six regressors/selector; native analytical routes and calibrated bridges.", evidence="Parent training-loop elapsed receipts cho 15 Graph +15 MLP cells; no separated test inference timers; regression score receipts thiếu stage timing.", finding="Có descriptive historical training-loop cost, không measured accuracy–deployment-cost frontier. Timer dừng trước outer-test prediction.", threats="Cache/checkpoint/validation overhead, unequal fit rows and budgets, concurrent CPU activity; sum-cell time không overall campaign wall time.", missing="Matched feature/transpilation/calibration/single-query and batch inference overhead, model load/cold start, deploy amortization.", impact="Material cho full RQ4 closure hoặc efficiency/deployment-superiority claim; không invalidates accuracy/negative OOD findings.", status="PARTIALLY ANSWERED", action="MINIMAL EXPERIMENTAL AMENDMENT", sources=["parent_runner","parent_protocol","profile","parent_protocol"], gaps="G01;G02"),
dict(rq="RQ4", track="B", question="Accuracy–estimation-cost/scaling trade-off trong từng simulator context là gì?", why="Microsecond execution targets có thể nhỏ hơn overhead estimator; planner cost có thể chi phối TN deployment.", minimum="Separate predictor preparation/inference/calibration/planning clocks, matched query timings and memory; target runtime scaling scored separately.", population="Seven predictive contexts, same-plan TN telemetry, 18 analog cells, historical dense/launch diagnostics.", target="Estimator overhead versus context-specific execution runtime; không so network build/path-search trực tiếp với warm contraction error.", split="Saved measurements and width/bond probes; no unified end-to-end estimator-cost protocol.", estimators="Graph/tabular/component predictors, TN native planner/EST, Pasqal unpromoted resource formula.", evidence="TN raw fields t_network_build_s/t_path_search_s/t_first_contract_s/t_warm_contract_s/t_end_to_end_s; historical kernel width/precision controls; no matched predictor latency panel.", finding="Target-runtime scaling có bounded evidence; estimator scalability/accuracy-cost ranking chưa có direct matched measurements. TN telemetry hỗ trợ reanalysis, không identify internal cost-model evaluation latency.", threats="Build/search may be shared prerequisites, cached paths and repeated query amortization; engine time versus API wall time; saturated chi32/64 for q<=9.", missing="Prediction/preparation/calibration cost per-context and amortized decision overhead; broad high-width/high-entanglement evidence.", impact="Material cho full cost RQ; không chặn current per-context error comparisons hoặc native same-plan conclusions.", status="PARTIALLY ANSWERED", action="MINIMAL EXPERIMENTAL AMENDMENT", sources=["tn_telemetry","tn","sim","joint_protocol","launch","width_study","analog"], gaps="G01;G02;G06"),
dict(rq="RQ5", track="A", question="Method nào áp dụng được với input/metadata/clock nào trước execution?", why="Unavailable input và incompatible semantics có thể làm score vô nghĩa khi deploy.", minimum="Pinned method/input/output identity, pre-execution availability, qualification/calibration, eligibility and failure coverage, matched baseline strength.", population="Qualified 4.515 cohort; 230 Qonductor observations không full original one-circuit inventory; QCEC 32 pairs/11 recipes.", target="Observed service target versus scheduled/native/reliability/nominal output clocks.", split="Method-source audit + same-row accuracy + frozen eligibility; selection recommendations conditional, không trained meta-selector.", estimators="Learned adaptations; original QCRE/Qiskit components; Hyb component; Scholten generalized proxy; service bridges and references.", evidence="Fidelity cards, 51/41/40 support audit, 257 transforms, snapshots and QCEC boundary.", finding="Native timing suitable cho schedule questions, service bridges require labels. Neural omits shots/backend inputs; regression uses shots/native counters. Original Scholten unavailable, proxy không original accuracy.", threats="Nominal calibration and reconstructed recipe/angles/layout; historical author tensor uncertain; native component không full-paper replication.", missing="Prospective deployment validation and guaranteed selector; exact historical circuits/calibration unavailable.", impact="Không chặn conditional method-applicability framework; không được claim exact paper reproduction hoặc universal practical optimum.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["registry","parent_protocol","profile","qcec","transforms","controls"], gaps="G03;G08;G09"),
dict(rq="RQ5", track="B", question="Lựa chọn estimator theo engine/API/precision/quality và known inputs như thế nào?", why="Full-state, sampling, contraction và analog simulation là different tasks; cần estimate đúng endpoint.", minimum="Execution identity, pre-execution feature availability, quality/configuration policy, coverage, native/adapter fidelity; no post-run oracle inputs.", population="Core 150 hashes and limited frontier, QCSim 143, ladder140, TN188 unique paired hashes, analog18cells.", target="Explicit per-context clocks; configured chi/cutoff known trước run, observed fidelity/bond/status không predictor inputs.", split="Frozen per-context predictions/protocols; same-plan pairing; selection decision evaluated separately from oracle rung runtime.", estimators="Graph/classical, Azizov adaptations, family-aware local adaptation, Maestro-style component, native TN, Pasqal diagnostic.", evidence="Registry, context-specific metrics, ladder target definitions and safety controls.", finding="Start from task/clock/required accuracy then eligible methods; retain quality violations/abstention. Maestro component không Composer; TN requires selected plan; Pasqal cannot enter predictive ranking.", threats="One machine/engine configurations and different survivor sets; no universal policy or production safety proof.", missing="Prospective multi-engine selector and high-width quality transfer; optional for estimator benchmark, not required to reproduce closed selector.", impact="Conditional applicability answered; cost-optimal/practically safe automatic selection chưa established.", status="ANSWERED WITH QUALIFICATIONS", action="CLOSE", sources=["registry","sim","joint_protocol","quality","tn","analog"], gaps="G03;G06;G09"),
]


CONTEXTS = [
    ("dense_fp32", "150 hashes q2–9", "warm sample FP32", "hash-grouped OOF", "Graph/MLP/RF/XGB/linear/work-Ridge", "C21", ["sim","sim_pairs","sim_slices"]),
    ("dense_fp64", "150 hashes q2–9", "warm sample FP64", "hash-grouped OOF", "Graph/MLP/RF/XGB/linear/work-Ridge", "C22", ["sim","sim_pairs","sim_slices"]),
    ("fixed_mps", "150 hashes;147 quality passes", "warm get_state FP64 chi16", "hash-grouped OOF", "Graph/Ridge/family residual/agnostic", "C23", ["sim","sim_pairs","quality"]),
    ("joint_mps", "140 hashes;840 measured/900 assigned rungs", "warm get_state + fidelity", "hash-grouped OOF", "conditioned/agnostic joint heads", "C26;C27", ["sim","quality","joint_protocol"]),
    ("family_mps", "140 hashes;840 measured/900 assigned rungs", "same-rung warm get_state + fidelity", "separate family-component holdout", "conditioned/agnostic joint heads", "C26;C27", ["sim","sim_pairs","joint_protocol"]),
    ("aer", "150 hashes q2–9", "warm noisy Aer FakeSherbrooke Opt1 shots1024", "hash-grouped OOF", "source/hybrid/transpiled GNN + five regressors", "C25", ["sim","sim_pairs","sim_producing"]),
    ("maestro", "143/150 hashes q2–9", "fresh-process QCSim FP64 engine execution shots1000 optimizer off", "hash-grouped OOF/fold-local calibration", "component/Ridge/Graph/median", "C24", ["sim","sim_pairs","registry"]),
    ("tn", "603/612 plan sessions;188 unique hashes,201 members", "selected-plan EST versus same-plan warm scalar contraction", "paired path/workspace/precision", "native RUNTIME_EST", "C28", ["tn","tn_telemetry","tn_scaling"]),
    ("analog", "18 analog cells;17 quality passes", "host MPSBackend.run", "descriptive grid, no predictive holdout", "Pasqal resource formula not promoted", "C29", ["analog","registry"]),
    ("historical_width", "208 rows/204 duration labels/68 circuits; separate RX pilot", "prepared PyTorch execution; separate CUDA-Q sample API", "historical circuit/width holdouts and RX width pilot", "HGB/work Ridge/empirical work fits", "C30", ["width_study","launch"]),
]


GAPS = [
    ("G01","G2","P1; essential for full RQ4","RQ4","A/B","Matched estimator computation overhead missing","Blocks full accuracy–cost and deployment-efficiency claims, not core predictive accuracy.","Stage and end-to-end cold/warm timings using frozen estimators/data; AM1 only after approval.","Request AM1 only if broad RQ4 must close; otherwise leave RQ4 PARTIALLY ANSWERED."),
    ("G02","G1","P1; main cost inventory closed here","RQ2/RQ4","A/B","Saved cost/scaling records need explicit timer boundaries and denominators","Does not block accuracy; can narrow AM1 but cannot supply missing inference timers.","Inventory timer boundaries; summarize 30 parent training receipts; TN phase telemetry and historical width study separately.","30-cell loop and 603-session TN summaries completed here; optional finer distribution analysis remains read-only, no new inference."),
    ("G03","G3","P1 breadth limitation","RQ1/RQ5","A/B","Not all originally listed papers have original-method predictive tests","Blocks claim complete paper-exact leaderboard: original Scholten0 eligible; Pasqal no predictive fit; Maestro component only.","Kernel/QV inputs or identified analog target; author software/labels if claiming original reproduction.","Disclose unavailable/diagnostic/adapted rows; do not revive methods merely to increase method count."),
    ("G04","G3","P1 domain limitation","RQ2/RQ3","A","Source/clock/workload/backend/shot confounding and no source-held-out study","Blocks isolated hardware-generation/source transfer, not archived pooled/source-balanced comparisons.","Crossed same-circuit contexts with compatible runtime boundaries; more training alone cannot recover missing historical linkage.","Defer; retain known/new structure scopes and source-balanced results."),
    ("G05","G3","P1 domain limitation","RQ1/RQ3/RQ5","A/B","Qualified reconstructed and resource-filtered populations not domain-representative","Blocks universal QPU/simulator conclusions, not bounded qualified-data results.","Verified broader inputs/contexts, not more repeated observations from same structures.","State observed distribution, structural support, qualification and terminal coverage."),
    ("G06","G3","P1 breadth limitation","RQ2/RQ4","B","Current-context high-width/cross-hardware transfer mostly unevaluated","Blocks general portable-simulator/scaling-law claims; historical PyTorch width result cannot fill another engine's gap.","Matched engine/task transfer with wider independent structures and resource feasibility; no such experiment needed for current local claims.","Defer beyond freeze; do not equate q2–16 manifest with q2–16 predictive accuracy."),
    ("G07","G3","P2 stronger claim","RQ2","A","Systematic unseen-shot and source transfer absent","Blocks those extrapolation claims; six observed shot values and shot multiplication do not validate transfer.","Shot/source-held-out crossed observations or new qualified labels; no retrospective observation pooling fixes confounding.","Defer; no proposed extra training for leaderboard improvement."),
    ("G08","G3","P2 mechanism/reproduction","RQ1/RQ3/RQ5","A/B","Unequal feature/capacity/refit budgets; author real tensor provenance unresolved","Blocks isolated topology causality and exact author execution claims; pipeline benchmark still legitimate.","Verified author tensor/config or matched equal-information controls; not necessary for benchmarking declared pipelines.","Keep51/41/40 and procedural asymmetry explicit. Do not hard-code41 or rerun to improve a rank."),
    ("G09","G3","P1 deployment limitation","RQ3/RQ5","A/B","No prospective runtime/quality guarantee","Blocks safe production scheduler or calibrated future-job coverage claim, not valid catastrophic-failure findings.","Prospective calibration/evaluation and decision loss with fixed quality policy, a separate research scope.","Keep conditional CIs, observed quality violations and abstention; no guarantee language."),
    ("G10","G0","P1 reporting, closed here","RQ1–RQ5","A/B","RQ numbers/target clocks and completion terminology can be confused","Could misstate research coverage without changing any experiments.","Explicit new-working-RQ to historical-authority crosswalk; per-track/context closure and units.","Closed in this package; old RQ4 files now serve new RQ2, no frozen renaming."),
    ("G11","G0","P1 reporting, already corrected","RQ3/RQ4","B","Old TN core/frontier prose label and topology aggregation ambiguity","Would misstate a local finding, not numerical results.","Use core3.734/frontier4.132 from saved table; historical topology mean predictions then score.","Preserve thesis correction; do not rewrite old artifacts."),
    ("G12","G1","P2 optional analysis","RQ3/RQ5","A/B","Application-specific acceptable-error thresholds not defined","No practically useful scheduling guarantee can be inferred from winner alone.","If a user supplies a decision tolerance, report frozen residual exceedance and quality failure rates without changing fits.","Optional saved-evidence analysis only; do not invent post-hoc success thresholds."),
]


def coverage():
    out = []
    for c in CARDS:
        if c["track"] == "A":
            out.append(dict(rq=c["rq"],track="A",context_id="qualified_qpu",dataset_unit=c["population"],runtime_target=c["target"],estimator_classes=c["estimators"],evaluation_split=c["split"],evidence=c["evidence"],status=c["status"],action=c["action"],finding=c["finding"],gap_ids=c["gaps"],source_paths=";".join(SOURCES[k] for k in dict.fromkeys(c["sources"]))))
            continue
        for ident,pop,target,split,models,claim,keys in CONTEXTS:
            status,action = c["status"],c["action"]
            if ident == "analog" and c["rq"] in ("RQ1","RQ2"):
                status,action = "UNANSWERED","DEFER WITH EXPLICIT LIMITATION"
            elif c["rq"] == "RQ2":
                status = "ANSWERED WITH QUALIFICATIONS" if ident in ("family_mps","historical_width") else "PARTIALLY ANSWERED"
                action = "CLOSE" if ident in ("family_mps","historical_width") else "DEFER WITH EXPLICIT LIMITATION"
            elif c["rq"] == "RQ4":
                status,action = "PARTIALLY ANSWERED","REANALYZE SAVED EVIDENCE" if ident in ("tn","analog","historical_width") else "MINIMAL EXPERIMENTAL AMENDMENT"
            notes = {
                'dense_fp32': 'RF MAE0.821ms; Graph1.716ms; local shared hashes, no current hardware transfer or measured predictor latency.',
                'dense_fp64': 'RF MAE0.813ms; Graph1.730ms; precision-specific fits, not precision-OOD experiment.',
                'fixed_mps': 'Graph MAE0.29609s versus Ridge1.27567s, saved paired CI negative; three quality failures remain separate.',
                'joint_mps': 'Family runtime increment unresolved;64/140 selected quality failures and ten unassessed; measured-rung conditioning explicit.',
                'family_mps': 'Frozen family-component holdout; conditioned/agnostic MAE0.31101/0.30879s, no resolved conditioning benefit.',
                'aer': 'Source/transpiled GNN MAE0.44966/0.30076s; transpiled XGB0.29704s; one fake backend/optimization level.',
                'maestro': 'Component MAE0.10370ms versus Ridge0.13361ms; CPU engine-reported time, not whole API or closed selector.',
                'tn': 'EST/warm median core3.734/frontier4.132 under same plan; build/search available as separate telemetry, no portable internal cost law.',
                'analog': '18-cell quality/time diagnostic; no identified/promoted predictor or predictive holdout;17quality passes.',
                'historical_width': 'PyTorch width holdout and RX precision/API pilot are bounded historical evidence; not common-core scores or estimator latency.',
            }
            out.append(dict(rq=c["rq"],track="B",context_id=ident,dataset_unit=pop,runtime_target=target,estimator_classes=models,evaluation_split=split,evidence=claim+"; matched context only",status=status,action=action,finding=notes[ident],gap_ids=c["gaps"],source_paths=";".join(SOURCES[k] for k in keys)))
    assert len(out) == 55 and len({(r['rq'],r['track'],r['context_id']) for r in out}) == 55
    return out


def csv_text(data):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(data)
    return stream.getvalue()


def readable(text):
    """Expand compact drafting labels in authored prose, never source files."""
    replacements={
        'cohort4515':'cohort 4.515', 'population/context':'population/context',
        'workingRQ':'working RQ', 'full21case':'full 21-case',
        'OldRQ4grid21/21':'Historical RQ4 grid 21/21', 'workingRQ2':'working RQ2',
        'workingRQ4overhead':'working RQ4 overhead', 'và12-part':'và 12-part',
        'từngRQ/Track':'từng RQ/Track', ':55context rows':': 55 context rows',
        'vàtopfivegaps':'và top five gaps', 'planningonly':'planning only',
        'đầy đủaccuracy–cost':'đầy đủ accuracy–cost', 'negativefindings':'negative findings',
        'RQ4estimator':'RQ4 estimator', 'OldRQ4':'Historical RQ4',
        'TrackA':'Track A', 'TrackB':'Track B', '30parentmanifests':'30 parent manifests',
        'Thirtyparentmanifests':'Thirty parent manifests', 'Currentcohort4515':'Current cohort 4.515',
        'sourcecounts340/230/3945':'source counts 340/230/3.945', 'groups165':'groups 165',
        '34metricrows':'34 metric rows', '49rows/sevenpredictivecontexts':'49 rows/seven predictive contexts',
        '51raw/41upstreamsim/40parent':'51 raw / 41 upstream sim / 40 parent',
        'RQ4fit-dependent':'RQ4 fit-dependent', 'Existing21case/126cell':'Existing 21-case/126-cell',
        '3.734/4.132':'3.734/4.132', 'overallmedian':'overall median',
        'New coverage55rows':'New coverage 55 rows', 'fiveTrackA':'five Track A',
        'fiveRQ×tenTrackBcontexts':'five RQ × ten Track B contexts',
        'sevenpredictive':'seven predictive', 'nativeTN':'native TN',
        'vàseparatehistoricalwidth':'và separate historical width', 'không55experiments':'không 55 experiments',
        'Contextstatus':'Context status', 'allowedactions':'allowed actions',
        'aszero':'as zero', 'timer/model':'timer/model',
        'RQnumbering':'RQ numbering', 'oldprotocols':'old protocols',
        'historical10/21':'historical 10/21', 'currentreport':'current report',
        'realtensor':'real tensor', 'originalmethod':'original method',
        'featurecausality':'feature causality', 'normalization/active-learning':'normalization/active-learning',
        'only21cases':'only 21 cases', 'thêm21cases':'thêm 21 cases',
        'TNphase':'TN phase', 'G1cost/scale':'G1 cost/scale',
        'globalwinner':'global winner', 'pooledwinner':'pooled winner',
        'QPUservice':'QPU service', 'analogrun':'analog run', 'mismatchedclock':'mismatched clock',
        'configuredchi':'configured chi', 'selectedplan':'selected plan',
        'Measuredfidelity':'Measured fidelity', 'observedbond':'observed bond',
        'targetruntime':'target runtime', 'historicallabels':'historical labels',
        'eligible sharedrows':'eligible shared rows', 'denseRF':'dense RF',
        'fixed-MPSGraph':'fixed-MPS Graph', 'Aercompiled':'Aer compiled',
        'GNNoverXGB':'GNN over XGB', 'Maestrocomponent scopeCPUengine':'Maestro component scope: CPU engine',
        'TNsame-plan':'TN same-plan', 'Pasqaldescriptive':'Pasqal descriptive',
        'qualityunmet':'quality unmet', 'measuredcost':'measured cost',
        'first/warm/precision':'first/warm/precision', 'Original0':'Original 0',
        '3399 eligible':'3.399 eligible', 'proxy3399':'proxy 3.399',
        'shared4515':'shared 4.515', 'full21 transfer':'full 21-case transfer',
        '18-cell':'18-cell', '|18-cell':'| 18-cell', '18cells':'18 cells',
        '230subset':'230 subset', 'observations liên kết32pairs/11recipes':'observations liên kết 32 pairs/11 recipes',
        'không230proofs':'không 230 proofs', 'QCEC:32zero-input':'QCEC: 32 zero-input',
        '8unitary':'8 unitary', '24counterexamples':'24 counterexamples',
        'groups165':'groups 165', 'QPack3945rows':'QPack 3.945 rows',
        'manifest204members':'manifest 204 members', 'q2–16/22families':'q2–16/22 families',
        '204independent':'204 independent', '150exact hashes':'150 exact hashes',
        'frontier42members':'frontier 42 members', 'Maestro143supported':'Maestro 143 supported',
        'ladder140assessed/150assigned':'ladder 140 assessed/150 assigned',
        '603/612plan sessions,201members/188hashes':'603/612 plan sessions, 201 members/188 hashes',
        'Analog18cells':'Analog 18 cells', 'dense208rows':'dense 208 rows',
        'RTX5070Ti/16GiB':'RTX 5070 Ti/16 GiB', 'RTX5070Ti':'RTX 5070 Ti',
        'Torch2.7.1':'Torch 2.7.1', '15manifest/method':'15 manifests/method',
        'All30manifest':'All 30 manifest', 'Source/clock/workload/backend/shot':'Source/clock/workload/backend/shot',
        '64/140 choices':'64/140 choices', 'core3.734/frontier4.132':'core 3.734/frontier 4.132',
        '3GiBdisk':'3 GiB disk', '2hours,1worker':'2 hours, 1 worker',
        '10queries/fold':'10 queries/fold', '30hashes/context':'30 hashes/context',
        '(50query identities)':'(50 query identities)', 'seeds42':'seeds 42',
        'ten warmups,30timed':'ten warmups, 30 timed', 'batch1/32':'batch 1/32',
        'query counts1/100/10000':'query counts 1/100/10.000',
        'Track A:4515parent':'Track A: 4.515 parent', 'denseFP32':'dense FP32',
        '2.7.1+cu128':'2.7.1+cu128', 'circuit job':'circuit job',
        'selectedmodel':'selected model', 'trainedcheckpoint':'trained checkpoint',
        'working RQ4overhead':'working RQ4 overhead', 'Currentcohort 4.515':'Current cohort 4.515',
        'ghi10/21':'ghi 10/21', 'authority21/21':'authority 21/21',
        '51raw':'51 raw', '40parent':'40 parent', 'oldRQ4':'historical RQ4',
        'completed21case':'completed 21-case', 'historicalRQ4':'historical RQ4',
        'medians3.734':'medians 3.734', '55context':'55 context', 'G0reporting':'G0 reporting',
        'G1saved-evidence':'G1 saved-evidence', 'G2new':'G2 new', 'G3scope':'G3 scope',
        'descriptive30cell':'descriptive 30-cell', 'campaignduration':'campaign duration',
        'inner1–3':'inner 1–3', 'inner0':'inner 0', 'chi32/64':'chi 32/64',
        'và840/900':'và 840/900', 'có603/612':'có 603/612', 'scored230':'scored 230',
        'tổng hợp15':'tổng hợp 15', 'tối đa10':'tối đa 10', 'tối đa30':'tối đa 30',
        'ceiling2':'ceiling 2', 'preserve3':'preserve 3', 'cells;17':'cells; 17',
        'MAE0.':'MAE 0.', 'Graph1.':'Graph 1.', 'Ridge0.':'Ridge 0.',
        '17quality':'17 quality', 'Old topology':'Historical topology',
    }
    for a,b in replacements.items():text=text.replace(a,b)
    return text


def framework():
    text = """# Khung câu hỏi và tiêu chí đóng nghiên cứu

## Working framework hiện tại

Mục tiêu là benchmark existing runtime estimators cho real QPU và simulator,
không phát triển neural architecture mới hoặc chứng minh Graph phải thắng.
Các RQ dưới đây là framework được người dùng yêu cầu cho assessment này.
Không sửa Research Contract v1.1 hoặc đổi tên frozen producing authority.

| Working RQ | Nội dung | Nối với historical authority |
| --- | --- | --- |
| RQ1 | Predictive accuracy/comparative effectiveness | Contract RQ1 pipelines, RQ2 analytical, RQ3 local simulator |
| RQ2 | Generalization | Contract RQ4; `artifacts/final_research/rq4_report.md` vẫn giữ tên cũ |
| RQ3 | Error/robustness/failures | Cross-cutting residual, support và quality analyses |
| RQ4 | Accuracy–cost/scalability | Target scaling evidence + estimator-overhead question; không phải old RQ4 grid |
| RQ5 | Fidelity/applicability/practical selection | Method/input/clock/quality identity, không new selection model |

Các headings trong Thesis Interpretation dùng numbering khác nữa. Khi trích
claim IDs C01–C32 phải giữ source ID và dùng crosswalk, không suy luận cùng
RQ number là cùng question. Historical contract còn ghi10/21; full-grid report
là current coverage authority21/21. Hai văn bản không được silently đồng bộ.

## Operational definitions

- ANSWERED: minimum evidence đầy đủ cho đúng population/endpoint đã nêu.
- ANSWERED WITH QUALIFICATIONS: có câu trả lời bảo vệ được nhưng phải giữ
  giới hạn về population, clocks, fidelity hoặc uncertainty.
- PARTIALLY ANSWERED: một phần quan trọng của broad question chưa evaluated.
- UNANSWERED: không có direct evidence cho question cụ thể, không phải vì
  method trả kết quả xấu. Pasqal predictive accuracy khác analog timing.
- CLOSE đóng bounded question, không cấp public-release approval.
- REANALYZE SAVED EVIDENCE không chạy model, checker hoặc workload mới.
- MINIMAL EXPERIMENTAL AMENDMENT là đề xuất có điều kiện, chưa authorized.
- DEFER WITH EXPLICIT LIMITATION phải nêu stronger claim nào bị cấm.

Unseen observation ID khác unseen consumed input, structural group, family,
backend, source và hardware. Exact hash grouping bảo vệ shared-circuit splits
nhưng không chứng minh tất cả near-duplicate structures độc lập. Backend
holdout được phép lặp cấu trúc có chủ ý; family holdout dùng structural embargo.
Case assignment không là independent observation. Ranking là conditional theo
test population, target, estimator information và training procedure.

Target-runtime scaling là thời gian simulator/QPU thực thi. Estimator overhead
là chi phí tạo estimate: parsing/features/transpile/planning/calibration/load/
prediction. Training là offline cost, không chia tổng training duration cho số
test rows để giả làm inference latency. Calibration intercept không physical
launch overhead. Native estimator availability có thể cần plan trước execution.

## Đối soát từng RQ và Track
"""
    labels = [("question","Câu hỏi"),("why","Vì sao quan trọng"),("minimum","Minimum evidence"),("population","Dataset và observation unit"),("target","Runtime target"),("split","Evaluation protocol"),("estimators","Estimators"),("evidence","Evidence hiện có"),("finding","Finding"),("threats","Threats to validity"),("missing","Bằng chứng thiếu"),("impact","Ảnh hưởng đến kết luận")]
    for c in CARDS:
        text += "\n### " + c["rq"] + " — Track " + c["track"] + "\n\n"
        for key,label in labels: text += "**" + label + ".** " + c[key] + "\n\n"
        metric_names = {
            'RQ1': 'Coverage, MAE/MedAE/RMSE, R², log1p/relative errors, p90/p99/max, skill versus matched references và paired MAE-delta intervals; analytical eligible subsets riêng.',
            'RQ2': 'Per-case MAE/MedAE/RMSE/R² và reference skill, shared-ID deltas, group support, known/new structure counts, input collision counts, train/test support distributions; không average R².',
            'RQ3': 'Signed residuals, over/underprediction counts, p90/p99/max, relative-error valid counts, per-source/group metrics, missingness/terminal/quality failure rates; EST/warm ratio ở TN.',
            'RQ4': 'Historical recorded loop min/median/max/sum và target-runtime width/bond errors có sẵn; matched estimator-stage/prediction latency, peak memory và amortized cost chưa có đầy đủ.',
            'RQ5': 'Eligible/assigned coverage, input/clock/fidelity class, qualification level, shared-row accuracy và quality violations; không có universal selection score hoặc deployment-cost optimum.',
        }
        text += "**Metrics và đơn vị.** " + metric_names[c['rq']] + "\n\n"
        text += "**Status:** " + c["status"] + ". **Action:** " + c["action"] + ".\n\n"
        text += "**Exact source paths:** " + refs(*dict.fromkeys(c["sources"])) + ".\n"
    return text


def completeness(f):
    profile=f["profile"]["source_profiles"]
    text="""# Benchmark completeness

## Kết luận theo mục tiêu

Core accuracy và observed-failure benchmark có thể bảo vệ được trong qualified
population/context hiện có. Original agenda rộng hơn chưa phủ hết. Đặc biệt
working RQ4 về estimator computational overhead chưa đủ matched measurements.
Reporting PASS không biến phần thiếu thành completed experiment. Ngược lại,
QNN/Random failures và no-resolved family/topology gain là valid answers.

## Method coverage

| Method/class | Đã thực sự evaluated | Còn thiếu / accepted boundary |
| --- | --- | --- |
| Ma–Li Track A | Graph+global và global-only adaptations, shared4515 OOF, full21 transfer grid | Không exact author tensor/normalization/active-learning reproduction; no topology-only causality |
| Qonductor | Sáu regressor families, pinned CX/CZ/ECR counter, 307 candidates, inner-only selector | Bounded search, mitigation not reproduced; không full upstream orchestration |
| QCRE/Tremba | Original pinned timing component + separately calibrated bridges | Nominal compiled unitary view, no full compilation workflow/historical routing reproduction |
| Qiskit | Native duration API + shot/calibrated variants | Schedule không observed service; measurement semantics khác QCRE |
| Scholten | Generalized depth/throughput proxy3399 eligible | Original0 verified eligible: missing kernel/QV semantics; proxy không chứng minh original predictive accuracy |
| Hyb-HANAS | Identifiable cost component, raw/log routes, matched eligible controls | Không full NAS/gradient/steps estimator; useful-work cost khác literal service |
| Simple/classical references | Train median/mean/source median and selector | Source median biết provenance; retrospective control không preregistered primary method |
| Ma–Li-style simulator | Graph/global/complexity controls per context | Không cùng full paper implementation cho mọi engine |
| Azizov | Source/hybrid/transpiled GNN + five regressors, noisy Aer core | Một fake backend/Opt1; authors' complete encoding unavailable |
| Family-Aware | Fixed-MPS residual and joint six-chi runtime/quality adaptation + family holdout | Không original Quantum Rings target; safety failures visible |
| Maestro | Fold-local CPU QCSim component, matched Ridge/Graph/median | Một engine/optimizer-off; không closed Composer hoặc full automatic selector |
| cuTensorNet | Native same-plan RUNTIME_EST paired with warm scalar contraction | Opaque internals; không sample/state/API wall-clock predictor |
| Pasqal |18-cell analog quality/time companion | Chưa identified/promoted resource predictor; không coi có table là predictive benchmark |

Các raw/affine/log-affine variants không phải từng paper độc lập. Native
component có thể original trong một overall nominal-context adaptation.
Dataset khác không tự làm implementation kém faithful; thay representation,
training/search/normalization/target mới là cần khai báo.

## Workload coverage

| Source | Rows / groups | Logical width min–median–max | Logical depth min–median–max | Shot coverage |
| --- | --- | --- | --- | --- |
"""
    for source,p in profile.items():
        ds=p['distributions']; w=ds['logical_allocated_width'];d=ds['logical_depth'];s=ds['shots']
        text+=f"| {source} | {p['rows']} / {p['groups']} | {w['min']:g}–{w['median']:g}–{w['max']:g} | {d['min']:g}–{d['median']:g}–{d['max']:g} | {s['min']:g}–{s['max']:g} |\n"
    text+="""
Ma–Li có source logical QASM; Qonductor có exact submitted physical QASM
và qualified logical recipes, không historical logical recovery. QPack là
angle-insensitive structural reconstruction, không original parameter/routing
recovery. 230 Qonductor observations liên kết32pairs/11recipes, không230proofs.
QCEC:32zero-input prepared-state proofs/output matches;8unitary proofs,
24counterexamples dưới assumed mappings; ZX inconclusive không non-equivalence.

Main structural groups165 không bằng số raw QASM hashes. QPack3945rows chỉ
sáu groups. Số rows lớn chủ yếu thêm repeated contexts/observations, không
phủ toàn circuit design space. Qualification filtering thay đổi workload/source
distribution; original Qonductor full inventory không thuộc scored230subset.

Simulator manifest204members q2–16/22families là attempt envelope, không
204independent successful predictive cases. Common core predictive scores dùng
150exact hashes q2–9; frontier42members q10–16 chủ yếu coverage/TN diagnostics.
Maestro143supported hashes; MPS ladder140assessed/150assigned hashes và840/900
rungs. Configured chi32/64 đã quá maximum Schmidt rank ở q<=9, không high-bond
asymptotic evidence. Native TN có603/612plan sessions,201members/188hashes.
Analog18cells không được gộp vào digital panel. Historical dense208rows là
separate target/corpus, không thêm vào current Track B denominator.

## Environment và evaluation conditions

- QPU gồm mười backend contexts; same-backend nominal snapshots không job-day
  calibrations. Device generation đi cùng source/workload/clock/shot shifts.
- Local GPU profile là RTX5070Ti/16GiB; contexts/software/precision/threads
  giữ riêng. Aer fake snapshot là noise context, không measured QPU execution.
- Main comparisons giữ shared outer-test IDs; neural/regression information,
  parameter capacity và fit population không hoàn toàn matched.
- Parent neural dùng inner1–3 fit/inner0 epoch selection, regression full
  outer refit; RQ4(old) neural cases có fresh full-outer refit. Không dùng
  case study score để sửa lại parent procedure sau khi thấy kết quả.
- Relative errors có denominator/valid-count riêng; low-support backend/size
  CIs không tự tạo. Conditional group bootstrap không future-job guarantee.
- Source-balanced versus pooled answer different estimands. Không tự promote
  pooled winner, không average R² và không pool overlapping transfer cases.

## Coverage của evaluation dimensions

| Dimension | Track A | Track B |
| --- | --- | --- |
| Accuracy/competitive controls | Shared rows, eligible controls, paired CIs | Matched context scores, paired comparisons; analog exception |
| Generalization | Nine family/ten backend/two small-support sizes; no isolated generation/source/shot transfer | Hash OOF, family MPS and historical width evidence; portable-engine claim absent |
| Failures/uncertainty | Tails, QNN/Random, collisions, source ranking, coverage | Quality safety, tails, TN bias, resource frontiers; no prospective guarantee |
| Target-runtime scaling | Two narrow QPack probes, observational size/shot data | Width/bond/precision studies conditional on engine/task |
| Estimator compute cost | Training-loop records only; no matched end-to-end/inference panel | TN stage telemetry available; no matched predictor-cost frontier |
| Fidelity/applicability | Source/target/qualification registry | Engine/API/config/quality registry; diagnostic/unsupported routes explicit |

## Chi phí đã có trong saved evidence

Các số sau chỉ tổng hợp15manifest/method của parent run trên RTX5070Ti,
Torch2.7.1+cu128. Timer gồm training/validation/cache/checkpoint loop overhead;
elapsed được chốt trước outer-test prediction. Không gọi là pure training-kernel
cost, inference latency hoặc total campaign duration. Model construction và
last checkpoint write nằm ngoài elapsed boundary; previous checkpoint writes
và validation/cache work thuộc loop. Đây không matched-cost
experiment và không dùng tỷ số để claim deployment speedup.

| Method | Cells | Min / median / max recorded loop seconds | Sum cell seconds |
| --- | ---: | --- | ---: |
"""
    for method,v in f['training_receipts'].items():
        text+=f"| {method} | 15 | {v['min_seconds']:.6f} / {v['median_seconds']:.6f} / {v['max_seconds']:.6f} | {v['sum_cell_seconds']:.6f} |\n"
    text+="\nProducing timer: "+refs('parent_runner')+". All30manifest paths/hashes ở provenance manifest.\n\n"
    text+="""Regression score receipts giữ params/scores/prediction pins nhưng không
tách inference timing. TN stage telemetry được tổng hợp dưới đây từ 603 successful
plan sessions (201 panel members, ba repetitions/member). Mỗi warm session lấy
median của năm warm contractions trước khi tổng hợp. Chín timeout sessions vẫn
thuộc denominator 612, không biến missing duration thành zero. Các median này
pool core/frontier, chỉ mô tả successful-session timing; không thay core/frontier
EST/warm ratios hoặc chứng minh independent structural uncertainty.

| TN stage | Successful sessions | Median seconds | Min–max seconds |
| --- | ---: | ---: | --- |
"""
    for k,v in f['tn_phases'].items():
        if isinstance(v,dict):text+=f"| {k} | {v['n']} | {v['median_seconds']:.9f} | {v['min_seconds']:.9f}–{v['max_seconds']:.9f} |\n"
    text+="""\nCột `t_end_to_end_s` có mặt nhưng không có giá trị ở cả 3.672 raw rows.
Không thể dùng table này để báo measured whole-API end-to-end time. Build/search
là prerequisite để có selected plan/EST; first/warm contraction là execution
cost. Không cộng các median độc lập thành median end-to-end, và không gọi path
search duration là isolated internal RUNTIME_EST evaluation latency. Đây là G1
read-only reanalysis đã hoàn tất, chưa lấp được G2 matched estimator-cost gap.
\n"""
    text+="Sources: "+refs('profile','parent_protocol','registry','sim','sim_widths','joint_protocol','tn_telemetry','qcec')+".\n"
    return text


AMENDMENT = """# Minimal experimental amendment — planning only

## AM1: Đo estimator overhead trên frozen estimators

**Status: NOT AUTHORIZED.** Một amendment, áp dụng hai tracks theo contexts,
không thêm model hoặc retrain để cải thiện leaderboard. Chỉ cần nếu thesis
muốn đóng đầy đủ RQ4 accuracy–cost; bounded accuracy/failure claims không cần.
Không suy ra phải chạy AM1 chỉ vì file này tồn tại.

**Hypothesis.** Khi tính parsing/features/transpilation/model loading và
prediction, một accuracy advantage nhỏ có thể không bù được estimation cost,
đặc biệt ở simulator targets dưới millisecond. Hypothesis này có thể bị bác:
matched end-to-end cost có thể negligible, hoặc amortize tốt với repeated queries.

**Exact question.** Trên cùng frozen query/input boundaries, incremental cost
của Graph/MLP/classical/analytical/component estimate là bao nhiêu, đổi lấy
held-out accuracy nào đã lưu? Không đo quantum execution trong amendment này.

**Tracks/data/unit.** Track A:4515parent, dùng từng frozen outer-fold checkpoint
và transform; predeclare tối đa10queries/fold, chọn label-free theo hash và
structural-size strata (50query identities), report size/group coverage.
Track B: tối đa30hashes/context chọn label-free từ shared core cho denseFP32,
fixed-MPS, Aer và QCSim, chỉ những contexts có đủ serialized estimators.
Đơn vị là estimator invocation tại declared input/config, không circuit job.
Các numbers này là proposed resource/sample limits, chưa frozen protocol.

**Target.** End-to-end estimator wall seconds, stage seconds, peak CPU RSS/
CUDA memory, model artifact bytes; offline training/calibration time riêng.
Existing quantum/simulator runtime labels chỉ dùng làm denominator/accuracy
reference, không remeasure. GPU synchronization trước/sau timed neural calls.

**Baselines.** Train-only median, frozen MLP, frozen selected classical model,
QCRE/Qiskit/raw-and-calibrated routes cùng query where eligible. Track B dùng
frozen matched Ridge/RF/component theo context; không force mọi model qua task
không phù hợp. Không choose comparator sau khi xem cost.

**Preparation gate trước execution approval.** Inventory serialization và
hashes. Không giả rằng predictions+params là trained checkpoint. Nếu classical
model chưa serialize, dừng route và xin duyệt bounded selected-model refit
trên original outer-train với original selected params; không rerun search,
không neural retrain, không thay frozen predictions hoặc score. Nếu không duyệt
refit, chỉ báo supported subset, không claim full cost comparison.

**Split.** Reuse outer-test identities và corresponding outer-trained models;
no tuning/inference rule fitted on test targets. Existing accuracy giữ nguyên.
Calibration reuse frozen coefficients; chỉ đo invocation, không chọn calibrator
theo latency. Preprocessing cost dùng circuits known before execution.

**Controls.** Same host/software/threads/precision, no concurrent workloads.
Một worker; predeclare cold process/model-load versus warm reuse; model-cache,
parsed/compiled input reuse và batch1/32 boundaries; ten warmups,30timed repeats
where feasible. Include parsing/features/target transpilation/planning if route
needs them, and separately show reusable-prepared-input scenario. Include
median-of-three cost, không time one seed rồi gán cho ensemble. Quality reference,
original simulator execution và measured telemetry không được predictor inputs.

**Metrics.** Median/p90/p99 latency and stage share per structural group/context,
coverage/failures, memory, ratio estimator latency / existing target runtime,
paired same-query cost deltas. Combine with existing MAE/relative errors and
conditional intervals, không compute new leaderboard accuracy from timed reruns.
Offline cost amortization for assumed query counts1/100/10000 phải ghi là
scenarios, không actual scheduler throughput hoặc measured economic gain.

**Expected computation/value.** No new GPU training, QPU job, simulator execution
or QCEC. New feature/planning/inference measurements are experiments and need
approval. Proposed ceiling2hours,1worker, preserve3GiBdisk floor and no automatic
resource growth. Đây là budget ceiling, không measured ETA. Existing cost logs
được kiểm trước để giảm số measurements. AM1 có thể thay recommendation về
expensive predictors cho tiny targets, không thay frozen accuracy/failure facts.

**Stop criterion.** Stop on identity/model/checkpoint/feature mismatch, missing
serialization, protocol ambiguity, cap exhaustion or inadequate disk/RAM.
Report incomplete coverage, không đổi dataset/mappings hay tune implementation
để có tốc độ đẹp. Stop once predeclared query/repeat grid terminates. Không
escalate thành simulator scaling sweep hoặc new training queue.

## Vì sao không đề xuất thêm experiments lúc này?

Source/hardware causal transfer cần crossed compatible labels, không chỉ train
thêm21cases. Original Scholten/Pasqal/closed Maestro cần khác target/input scope.
Topology-only gain không mục tiêu chính của benchmark existing pipelines.
Các câu hỏi này được ghi G3, không lén yêu cầu rerun để tăng model count hoặc
đẹp scores. G1cost/scale reanalysis và target-threshold analysis không cần
experiment nhưng chỉ dùng evidence/decision thresholds có thật.

Nếu không muốn mở freeze, lựa chọn khoa học hợp lệ là giữ RQ4 PARTIALLY ANSWERED
và không claim deployment efficiency. Không tự đổi nó thành ANSWERED bằng
wording hạn chế đến mức bỏ qua câu hỏi chi phí.
"""


def render():
    pins=load_sources(); f=facts(); cov=coverage()
    files={"rq_framework.md":framework(),"benchmark_completeness.md":completeness(f),"minimal_amendment_proposals.md":AMENDMENT}
    files['rq_coverage_matrix.csv']=csv_text(cov)
    md="# RQ coverage matrix\n\nTen Track×RQ summary decisions dưới đây đánh giá broad working framework.\n55context rows trong CSV giữ target, population, split, estimators và exact source\npaths; historical width/analog/native-plan contexts không nhập vào seven predictive\ncontexts. Status không được tính từ test PASS hoặc average context scores.\n\n| Working RQ | Track A | Track B | Recommended action |\n| --- | --- | --- | --- |\n"
    for n in range(1,6):
        a,b=[c for c in CARDS if c['rq']==f'RQ{n}']
        md+=f"| RQ{n} | {a['status']} | {b['status']} | A: {a['action']}; B: {b['action']} |\n"
    md+="\n## Exact context coverage\n\n| RQ | Track/context | Unit / target | Split | Status | Evidence |\n| --- | --- | --- | --- | --- | --- |\n"
    for r in cov:
        sources="; ".join('['+Path(p).name+']('+str(ROOT/p)+')' for p in r['source_paths'].split(';'))
        md+=f"| {r['rq']} | {r['track']}/{r['context_id']} | {r['dataset_unit']} / {r['runtime_target']} | {r['evaluation_split']} | {r['status']} | {sources} |\n"
    md+="\n[Full twelve-part analyses](rq_framework.md) giải thích minimum evidence,\nfindings, threats và missing evidence từng track. Native TN không train/test\nestimator model; same-plan check là evaluation phù hợp cho native estimator.\nPasqal RQ1/RQ2 UNANSWERED không phủ nhận analog quality/time measurements.\nHistorical width RQ2 qualified chỉ đóng local-width question, không current\nCUDA-Q/Aer cross-engine transfer.\n"
    files['rq_coverage_matrix.md']=md
    md="# Scientific gap register\n\nG0reporting; G1saved-evidence analysis; G2new experimental evidence;\nG3scope/input limitation. Priority không là execution order. Không có P0\ninvalidating frozen bounded accuracy claim; có material G2 cho full workingRQ4.\n\n| ID | Class / priority | RQ / Track | Gap | Blocks which claim? | Smallest evidence | Action |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    for g in GAPS:
        i,cl,priority,rq,tr,desc,impact,evidence,action=g
        md+=f"| {i} | {cl}; {priority} | {rq}; {tr} | {desc} | {impact} | {evidence} | {action} |\n"
    md+="""\n## Top five theo scientific impact

1. G01: estimator accuracy–cost thiếu direct matched measurements. Material
   cho broad RQ4, không inference speed claim từ parameter count.
2. G03: original-method breadth. Một số original objectives chỉ đạt adaptation,
   component hoặc unavailable; phải giữ trong comparison envelope, không gọi
   danh sách paper là số full reproductions đã test.
3. G04/G05: source/workload/clock confounding và limited independent structures.
   Đây là giới hạn suy luận, không lỗi pipeline đã phát hiện. More rows không
   tự sửa được, và claims phải conditional theo observed distribution.
4. G06: simulator OOD/support ngoài small-width local contexts chưa phủ đủ.
   Historical width evidence có giá trị nhưng không thay thế direct test trong
   engine/context khác. First/warm/precision slices không cross-hardware transfer.
5. G08/G09: unequal pipeline information/procedure và prospective applicability.
   Không infer isolated topology gain, source-blind hardware adaptation hoặc
   safe scheduler từ pooled errors/conditional CIs.

G02 đã bổ sung descriptive30cell loop-cost inventory và 603-session TN phase
summary trong completeness report. Có thể phân tích finer distributions nếu
cần; không đổi question của overhead thành target-simulator runtime để tránh gap.
G10/G11 đã có wording closure, không dùng để chứng minh experimental completeness.
"""
    files['scientific_gap_register.md']=md
    files['research_decision_memo.md']="""# Research decision memo

## Verdict: PARTIALLY SUFFICIENT

**Core benchmarking completeness:** đủ bảo vệ comparisons của declared
estimators trên qualified QPU cohort và matched local simulator contexts,
cùng observed negative findings. Không hoàn tất mọi paper-exact objective.

**Broader research completeness:** chưa đóng toàn bộ năm workingRQ. RQ2 vẫn
partial ở source/shot/hardware/size/engine transfer. RQ4estimator computational
overhead có material measurement gap. Không đánh đồng completed21case grid
của historicalRQ4 với completed accuracy–cost question hiện tại.

**Publication readiness:** không đánh giá/duyệt release ở task này. Existing
metadata/rights/controlled-export gates giữ nguyên, không giải bằng experiment.

## Có thể đóng ngay

- RQ1A/B: conditional accuracy/comparator findings, không universal ranking;
  TrackB giữ Pasqal predictive exception và Maestro component boundary.
- RQ3A/B: actual failures/tails/source effects/collisions/quality violations.
  QNN/Random thất bại là scientific answers, không unanswered hypotheses.
- RQ5A/B: input/clock/fidelity applicability và conditional selection rules,
  không automated cost-optimal/safe production policy.

## Chưa đóng broad question

- RQ2A/B: completed observed transfers không isolate hardware generation;
  no source/shot-transfer guarantee; current simulator context portability
  untested. G3 không cần thêm model để khắc phục ngay.
- RQ4A/B: target scaling và historical training loop duration không cung cấp
  full estimator overhead frontier. Có partial evidence, không efficiency claim.

Không mâu thuẫn với previous READY WITH LIMITATIONS: verdict trước xét
defensibility của bounded findings, verdict này xét coverage năm questions
rộng hơn. Reporting verification chỉ kiểm fidelity/consistency, không bỏ phiếu
cho research sufficiency. Không numerical contradiction mới trong frozen scores.

## Practical method-selection framework

1. Chọn exact endpoint trước: QPUservice, schedule, simulator sample/state,
   engine component, scalar contraction hoặc analogrun. Reject mismatchedclock.
2. Kiểm input trước execution: logical/physical circuit tier, shots/native
   counters, nominalTarget/durations/errors, configuredchi, hoặc selectedplan.
   Measuredfidelity/observedbond/targetruntime không pre-execution features.
3. Không có train/calibration labels: native schedule/EST chỉ áp dụng đúng
   native target; không tự cộng physical overhead hoặc gọi service prediction.
4. Có suitable historicallabels: strong constant/classical controls trước;
   compare eligible sharedrows, sources/groups and uncertainty. Graph pooled
   gain nhỏ hơn trước selector và reversed khi equal-source weighting.
5. Simulator: denseRF có local accuracy; fixed-MPSGraph có signal; Aercompiled
   views useful nhưng GNNoverXGB unresolved; Maestrocomponent scopeCPUengine;
   family selectors không quality-safe, TNsame-plan only, Pasqaldescriptive.
6. OOD/collisions/qualityunmet: công bố limitation/abstention, không choose
   method chỉ vì pooledwinner. Chưa có measuredcost đủ để tối ưu deploy.

## Next decision

**Giữ nguyên freeze; chỉ phân tích saved evidence thêm khi có câu hỏi cụ thể.**
30-cell loop inventory và 603-session TN phase summary đã hoàn tất trong review
này, vẫn không tạo ra missing inference measurements. Không fit/calibrate/execute
model mới. Quyết định cần chủ dự án chọn:

- Giữ freeze, nhận RQ2/RQ4 partial và đóng bounded benchmark; hoặc
- Duyệt riêng AM1 nếu full accuracy–cost là required thesis contribution.

Không đề xuất rerun neural models để tăng score, thêm source labels giả lập,
revive old queue hoặc sửa split sau khi thấy failures. AM1 cần approval mới;
không hành động tiếp chỉ vì review file đã lưu. Không manuscript/Canva/release
work ở bước này.
"""
    files['README.md']="""# Research question closure

**Verdict: PARTIALLY SUFFICIENT.** Frozen benchmark có đủ evidence cho bounded
accuracy, observed-failure và applicability findings. Năm working questions
rộng hơn chưa được trả lời đầy đủ, đặc biệt estimator accuracy–cost overhead.
Scientific Freeze v2, cohort4515 và original results/protocols giữ nguyên.

| Working question | Track A | Track B |
| --- | --- | --- |
| RQ1 accuracy | ANSWERED WITH QUALIFICATIONS | ANSWERED WITH QUALIFICATIONS |
| RQ2 generalization | PARTIALLY ANSWERED | PARTIALLY ANSWERED |
| RQ3 failures/robustness | ANSWERED WITH QUALIFICATIONS | ANSWERED WITH QUALIFICATIONS |
| RQ4 accuracy–cost/scaling | PARTIALLY ANSWERED | PARTIALLY ANSWERED |
| RQ5 fidelity/applicability | ANSWERED WITH QUALIFICATIONS | ANSWERED WITH QUALIFICATIONS |

Đây là interpretation/closure package, không benchmark version mới hoặc
research amendment đã thực thi. OldRQ4grid21/21 là evidence cho workingRQ2,
không cho workingRQ4overhead. Context exceptions ghi riêng trong coverage matrix.

## Đọc theo thứ tự

1. [Decision memo](research_decision_memo.md): đóng được gì và cần quyết định gì.
2. [RQ framework](rq_framework.md): minimum evidence và12-part analysis từngRQ/Track.
3. [Coverage matrix](rq_coverage_matrix.md) / [CSV](rq_coverage_matrix.csv):55context rows.
4. [Completeness](benchmark_completeness.md): method/workload/condition/evaluation coverage.
5. [Gap register](scientific_gap_register.md): G0–G3 vàtopfivegaps.
6. [AM1 proposal](minimal_amendment_proposals.md): planningonly, NOT AUTHORIZED.
7. [Validation](validation_report.md) / [provenance](provenance_manifest.json).

Recommended next decision: giữ freeze với RQ2/RQ4 partial, hoặc duyệt riêng
AM1 nếu muốn đóng đầy đủaccuracy–cost; không cần experiment để giữ bounded
accuracy/negativefindings. Không commit/push, external workspace hoặc release.
"""
    # Numerical evidence and a finite provenance catalog are generated, not retyped.
    c=f['claims']
    numeric={"graph_mae_seconds":c['C01']['graph_mae'],"skill_vs_median":c['C01']['skill'],"skill_vs_selector":c['C02']['skill'],"graph_minus_selector_ci":[c['C02']['ci_low'],c['C02']['ci_high']],"tn_EST_over_warm_medians":f['tn_ratios'],"parent_recorded_loop_costs":f['training_receipts'],"tn_successful_session_stage_costs":f['tn_phases']}
    val="""# Validation report

Scope: saved-evidence analysis/reporting only. Không experiment, model inference,
new calibration, QCEC, dataset/protocol/checkpoint change hoặc historical rewrite.
Assessment judgement khác mechanical verification: PARTIALLY SUFFICIENT không
được đổi thành complete chỉ vì các checks bên dưới PASS.

## Numerical/source checks

- Currentcohort4515, sourcecounts340/230/3945, groups165 khớp frozen profile.
- TrackA34metricrows; TrackB49rows/sevenpredictivecontexts, không cross-trackpool.
- 51raw/41upstreamsim/40parent inputs giữ đúng support evidence; RQ4fit-dependent.
- Existing21case/126cell authority giữ nguyên, no case resume hoặc new refit.
- Graph skill ratios tính lại từ frozen MAEs; paired CIs lấy đúng saved direction.
- TN core/frontier medians3.734/4.132 lấy từ saved stratum rows; không overallmedian.
- Thirtyparentmanifests pin exact cells; elapsed timer precedes test inference.
  Cost inventory không là matched deployment timing hay campaignduration.
- New coverage55rows:fiveTrackA +fiveRQ×tenTrackBcontexts, gồm sevenpredictive
  contexts, nativeTN, analog vàseparatehistoricalwidth. Đây không55experiments.
- Contextstatus labels vàallowedactions được kiểm tra riêng; negativefindings
  không bị đổi thành UNANSWERED. No unavailable value imputed aszero.

Exact headline values used:

```json
"""+json.dumps(numeric,ensure_ascii=False,indent=2,sort_keys=True)+"\n```\n\n"
    val+="""## Integrity and limits of validation

Manifest pins all directly consumed source files,30parentmanifests, reporting
builder and originalrequest. Validation rechecks each source digest and exact
byte rendering of nine requested files; links must resolve and CSV rows/columns
must replay. Existing thesis validator is also re-run against its pinned
frozen-evidence sources/receipt, not treated as new experimental reproduction.
No timestamp/elapsed from execution is relabeled inference overhead.

No new primary-score contradiction found. Existing reporting clarifications
(TNlabels and historical mean-predictions scoring) remain. Frozen contract's
historical10/21 statements do not replace full21case currentreport. NewRQnumbering
is explicitly mapped, not silently applied to oldprotocols.

Notverified: author executed realtensor, universal originalmethod fidelity,
prospective safety, isolated generation/featurecausality or matched estimator
end-to-end cost. Absence of these prevents stronger claims. This validator
does not prove completeness, training reproducibility or publication approval.

Run from repo root (read-only):

```sh
PYTHONDONTWRITEBYTECODE=1 /home/server/Documents/.venv-qonductor/bin/python -B \
 benchmark_v1/scripts/build_research_question_closure.py
```

## Verification observed during this assessment

- Closure source/byte/numerical/link validator: PASS, 77 source pins,
  55 context rows, zero broken local links, zero authorized amendments.
- Focused tests: 36 passed in 6.25 s. Command used `python -B -m pytest -q
  -p no:cacheprovider` on `test_research_question_closure.py`,
  `test_thesis_interpretation.py`, `test_benchmark_interpretation.py` and
  `test_final_scientific_reporting.py`, with OPENBLAS/OMP threads set to 1
  and PYTHONDONTWRITEBYTECODE=1. These are reporting checks, not experiments.
- Existing thesis validator rerun: PASS, 25 reporting files and 66 source pins;
  historical thesis manifest remains
  `9a7744bae230928bde433ee0cc4064a9e4feaf89b060f5f9de1e78c0c8107c60`.
- `git diff --check`: exit 0. Existing 13 tracked modifications were already
  present before this task and were not edited. New analysis files only.

The test run above preceded this verification paragraph and final prose-spacing
edits. The same focused tests and both validators are rerun on final saved files
before handoff; the final response records that final run separately.

Sources below are exact local evidence paths/hashes, not a public-data export:

| Source | SHA-256 |
| --- | --- |
"""
    for path,digest in sorted(pins.items()):
        val+="| ["+str(Path(path).relative_to(ROOT))+"]("+path+") | `"+digest+"` |\n"
    files['validation_report.md']=val
    files={name:(readable(text) if name.endswith('.md') else text).rstrip()+"\n" for name,text in files.items()}
    request=Path('/home/server/.codex/attachments/48a36bac-8fcf-471b-af29-ed9f38a732aa/Pasted text.txt')
    pins[str(request)]=sha(request)
    pins[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    test_path=ROOT/'benchmark_v1/tests/test_research_question_closure.py'
    pins[str(test_path)]=sha(test_path)
    manifest=dict(scope="ANALYSIS_AND_PLANNING_ONLY",verdict="PARTIALLY SUFFICIENT",scientific_freeze="Final Scientific Freeze v2",source_pins=pins,payload={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},closure_decisions=[{k:c[k] for k in ['rq','track','status','action']} for c in CARDS],coverage_rows=55,experimental_amendments=[dict(id="AM1",authorized=False,necessary_for="full working RQ4 cost closure, not bounded accuracy claims")],new_training=False,new_inference=False,new_workloads=False,protocol_or_result_changes=False,commit_push_external_edits=False,self_hash_policy="manifest excluded from its own payload digest")
    files['provenance_manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    return files,manifest


def validate():
    files,manifest=render()
    for name,content in files.items():
        assert (OUT/name).read_bytes()==content.encode(), 'replay:'+name
    for path,digest in manifest['source_pins'].items():
        assert sha(path)==digest,'source_changed:'+path
    for name,content in files.items():
        if not name.endswith('.md'):continue
        for url in re.findall(r'\]\(([^)]+)\)',content):
            if url.startswith(('http://','https://','#')):continue
            p=Path(url.split('#')[0])
            assert (p if p.is_absolute() else OUT/p).exists(),(name,url)
    for c in CARDS:
        assert c['status'] in ('ANSWERED','ANSWERED WITH QUALIFICATIONS','PARTIALLY ANSWERED','UNANSWERED')
        assert c['action'] in ('CLOSE','REANALYZE SAVED EVIDENCE','MINIMAL EXPERIMENTAL AMENDMENT','DEFER WITH EXPLICIT LIMITATION')
    actual=list(csv.DictReader((OUT/'rq_coverage_matrix.csv').open(newline='')))
    assert actual==coverage()
    return dict(status="PASS",scientific_verdict="PARTIALLY SUFFICIENT",scope="source/byte/numerical/link verification; not experiment completeness proof",requested_files=9,provenance_files=1,source_pins_checked=len(manifest['source_pins']),context_rows=55,track_rq_decisions=10,material_G2_amendments_proposed=1,amendments_authorized=0,broken_local_links=0,scientific_freeze_v2_preserved=True,no_training_inference_simulation_QCEC=True,no_commit_push_external_edits=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-patch',action='store_true')
    parser.add_argument('--file')
    args=parser.parse_args()
    if args.emit_patch:
        files,_=render();files={args.file:files[args.file]} if args.file else files
        print('*** Begin Patch')
        for name,content in sorted(files.items()):
            path=OUT/name
            if path.exists():
                assert path.read_text()==content,'refuse_overwrite:'+str(path)
                continue
            print('*** Add File: '+str(path))
            for line in content.splitlines():print('+'+line)
        print('*** End Patch')
    else:
        print(json.dumps(validate(),ensure_ascii=False,indent=2,sort_keys=True))
