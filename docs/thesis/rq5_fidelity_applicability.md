> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# RQ5 — Method fidelity, applicability and practical selection

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

Ma–Li raw 51 gồm 44 gate counters,2 size và 5 structural fields. Support-based refinement cho original simulator corpus 41, main local QPU 40; không lấy 41 vị trí đầu. Current Dense adapter dùng 7 globals, không ép 41 lên mọi simulator. Historical executed real-QPU tensor của tác giả chưa rõ; source-native terminology không thay evidence. Graph/MLP hiện tại là architecture/core adaptations.

Practical selection trước hết hỏi label boundary, input availability, preprocessing compatibility và quality. Native scheduled duration chưa phải observed service; TN scalar contraction chưa phải CUDA-Q sample; MPS quality policy là phần của task. Không universal winner chỉ từ MAE.

## Limitation

QCEC 32 zero-input state/output agreements chỉ hỗ trợ existing recipe qualification.8 full-unitary PROVED và 24 COUNTEREXAMPLE dưới assumed mappings;230 observations liên kết 32 pairs, không 230 independent proofs. Không exact historical recovery hay compilation history. Registry fidelity và scope limits phải đi cùng score.

## Thesis takeaway

RQ5 **ANSWERED WITH QUALIFICATIONS**. Applicability/fidelity rules đóng được trong benchmark scope, không cần production-safe universal selector để thành một benchmark có giá trị.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); [registry](../../results/interpretation/thesis/estimator_fidelity_registry.md); [taxonomy](../../results/interpretation/thesis/tables/estimator_identity.csv); [feature](../../results/interpretation/benchmark/mali_feature_schema_audit.md); [qcec](../../results/reproducibility/qcec_report.md); [claims](../../results/research_questions/closure/final_claim_matrix.csv).
