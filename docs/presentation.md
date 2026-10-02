# Nội dung khoa học cho báo cáo benchmark

Phạm vi: báo cáo nội bộ khoảng 10 phút, tiếng Việt. Đây là nguồn nội dung duy
nhất cho deck; không tạo một outline/version song song. Số liệu QPU lấy từ
`benchmark_v1/execution/manifests/report_finalization.json` và artifact mà nó
pin. Kết quả predictor simulator E1–E6 lấy từ
`artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json`
và method cards tại đó; report manifest cũ chưa bao gồm các run này.
[Scientific review](scientific_review.md) chốt bảng phương pháp và giới hạn
claim hiện tại. Không training, đo thêm hoặc thử seed mới trong queue dựng báo cáo.

## Thông điệp chính

Benchmark gồm hai domain: dự đoán thời gian quan sát của real QPU từ dữ liệu
đã lưu, và dự đoán runtime simulator trên máy local. Giữ cùng ledger/split
trong từng phép so sánh, nhưng không gộp những clock khác nhau thành một bảng
xếp hạng. Có kết quả đủ để báo cáo; coverage của toàn bộ method matrix còn
PARTIAL. Không dùng từ “hoàn tất mọi replication”.

## Findings được phép trình bày

### 1. Graph adaptation có MAE thấp hơn polynomial trên các shared rows, không phải thắng mọi metric

Hai model được train/evaluate trên cùng unified ledger và frozen grouped
outer folds. Các dòng theo nguồn dưới đây là breakdown của cùng OOF run,
không phải train riêng trên Ma–Li/Qonductor/QPack.

| Nguồn | Shared test rows | Graph V3-large MAE (s) | Polynomial MAE (s) | 95% grouped-bootstrap CI của chênh MAE graph − poly (s) |
| --- | ---: | ---: | ---: | --- |
| Ma–Li | 340 | 0.596772 | 0.646791 | [−0.153046, +0.049747] |
| Qonductor | 4.481 | 1.027606 | 1.394624 | [−0.430236, −0.300482] |
| QPack | 3.945 | 0.943547 | 1.221289 | [−0.486787, −0.074311] |

Nguồn: `artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2/pairwise_comparisons.csv`;
lọc `candidate_method_id=mali_graph_architecture_v3_large`,
`reference_method_id=unified_polynomial_v3`, theo `source_id`.
Shared-row hashes, theo thứ tự bảng:

- `3ad175f3b2cf88f52f9f0dedd404684fbff696b2d776f8d55538083dcdeac8e7`
- `a785df4abc98e96ad899bdf4f53837806715232d440bc620de74b88639c74655`
- `d8f5438afbc5179bc99a96672022627411c6d6ead8597fcee6b9abbe383a36d7`

Giới hạn bắt buộc: CI Ma–Li đi qua 0; Qonductor có coverage bất đối xứng vì
graph không dự đoán row4477. Đây là paired descriptive result, không phải
primary superiority trên cả 4.482 rows. Graph có bảy global features, polynomial
có năm; không quy chênh lệch cho riêng graph structure. Cả hai là adaptation,
không phải reproduction nguyên bản của paper.

### 2. MAE tốt hơn chưa có nghĩa giải thích tốt biến thiên runtime trên QPack

Trên 3.945 QPack observations, R² graph V3-large = −0.094887;
polynomial = −0.631629. Dù graph có MAE thấp hơn, cả hai R² đều âm.
R² âm nghĩa tổng squared error lớn hơn việc dùng mean của đúng tập đánh giá
đó; mean này chỉ là tham chiếu toán học, không phải một deployable baseline
được fit từ test labels. Không nói model tốt theo mọi tiêu chí.

Nguồn: `unified_method_comparisons_v2/source_metrics.csv`, hai method trên,
`source_id=qpack_mcp`, `n=3945`, `metric_scope=row-weighted source slice`.
QPack có 46 workflows; dùng thêm equal-workflow MAE như diagnostic để tránh
workflow dài chi phối row-weighted metric. Workflow holdout không đồng nghĩa
unseen QAOA structure: sáu templates lặp lại trong nhiều workflows.

### 3. Scheduled duration không thay thế observed QPU service/execution time

QCRE snapshot critical path và Qiskit native duration có 8.766 dự đoán,
row4477 unavailable. Chúng là scheduled single-shot proxies; không bao gồm
mọi provider overhead. Việc hai route gần/đúng bằng nhau dưới cùng Target
là cross-check schedule, không phải hai independent model wins. Shot scaling
và affine/log-affine fit outer-train-only là các adaptation riêng; phải hiện
method-output clock bên cạnh evaluation-target clock.

Nguồn: `unified_method_comparisons_v2/method_metrics.csv`, `attempts.csv`,
fidelity registry v2. Các table final giữ raw và calibrated variants riêng.
Scholten nominal có 5.769 successful; Hyb repair có 8.420 finite predictions
nhưng raw score có thể overflow. Không suy đủ prediction là đủ accuracy.

### 4. Simulator chạy xong chưa đủ điều kiện đưa vào accuracy benchmark

Panel digital có 204 members / 191 exact hashes; core 162, frontier 42.
CUDA-Q MPS có 612 attempted session-cells mỗi clock: 573 ok, 21 quality_failed,
18 adapter_error. cuTensorNet có 612 attempted sessions: 603 ok, 9 timeout.
Đây là session-cell counts, không phải 612 circuit khác nhau. Không bỏ failures
khỏi coverage denominator. MPS cần quality gate; TN native estimate chỉ so
với contraction cùng plan, không phải toàn bộ wall-clock của simulator khác.

Nguồn: `artifacts/benchmark_v3/simulator/simulator_reader_pack_v1/manifest.json`
và matching configuration tables. C3a Aer graph diagnostic trên 162 core
rows/150 hashes có MAE 0.387654 s, R² 0.155835;
không gọi đó là Azizov GNN hoặc full-panel Ma–Li reproduction. Measurement
coverage của CUDA-Q không phải predictor coverage.

### 5. Maestro pilot cung cấp evidence về stability, không cung cấp predictor accuracy

S9 hoàn tất 10 synthetic calibration cells, 240 API calls: 90 untimed,
150 timed. Tất cả calls transport ok; 9 SV n16 cells không đạt frozen
timing-stability gate, 1 MPS n12/chi32 cell đạt. Trong cùng SV 1Q/n16/R256
cell, 15 reported timings trải từ 0.014905151 đến 7.355305934 s. MPS cell có
ba session medians 0.033975028 / 0.033144007 / 0.035826843 s.

Nguồn: `artifacts/benchmark_v3/simulator/maestro_candidate_runtime_resume/calibration/raw_records.csv`,
`run_manifest.json`, `checkpoint.jsonl`, `fitted_calibration.json`;
final builder tự tính lại median/MAD theo ngưỡng prospective relative 0.2,
absolute 50 µs. Ngưỡng là local engineering QA, không phải mức significance
hoặc ngưỡng do Maestro paper công bố.

Status `pilot_gate_failed`, không resume. 432 assigned calibration cells:
10 completed, 389 eligible unstarted, 33 quarantined. Panel 204 members/
408 candidate cells đã có parser/prediction preflight nhưng chưa có phép đo
runtime held-out tương ứng; không có accepted predictor/MAE score.
Clock là reported execution trong fresh process, không gọi warm persistent
process. Chưa xác định nguyên nhân variation; không đổ lỗi cho threading,
hardware hay thuật toán khi chưa có intervention evidence. Không suy Maestro
paper predictor thất bại trên common panel từ pilot này.

### 6. Aer: compiled-view GNN tốt hơn source-view trên cell đã test; so với XGBoost chưa có kết luận

Trên cùng 150 exact-QASM hashes, source/hybrid/transpiled GNN có MAE
0.4497 / 0.4064 / 0.3008 s. Transpiled XGBoost có MAE 0.2970 s.
Chênh source − transpiled GNN là +0.1489 s, khoảng 95%
[+0.0284,+0.3039]; chênh transpiled GNN − XGBoost là +0.0037 s,
khoảng [−0.1726,+0.1324]. Cả ba GNN dùng median predictions của seeds
42, 1234, 31415 và năm outer folds; không chọn seed tốt nhất.

Nguồn: E6 `aer_azizov_hash_metrics.csv` và
`aer_azizov_paired_bootstrap.csv`. Đây là local Azizov-style adaptations
trên một FakeSherbrooke/Opt1 context, không phải reproduction toàn scope paper.
Các views có feature contract khác nhau; không quy kết riêng topology graph.

### 7. MPS: graph có kết quả local tốt; lợi ích bổ sung từ family chưa được xác lập

Fixed CUDA-Q MPS FP64 bond-16 có 150 assigned hashes, 144 runtime labels hữu
hạn, 142 quality-pass và sáu unavailable. Trên 144 finite labels, graph MAE
0.3024 s; family residual 1.0009 s; family-agnostic ablation 1.0813 s.
Residual − ablation = −0.0804 s, khoảng [−0.2852,+0.0804] đi qua zero.
Trên 142 quality-pass labels, graph MAE 0.3037 s nhưng max error 25.0964 s.

Nguồn: E6 `mps_fixed_chi16_hash_metrics.csv`,
`mps_fixed_chi16_paired_bootstrap.csv` và E3
`mps_quality_pass_metrics_v1.json`. Các model được train cùng split và target;
quality-pass là lát đánh giá của model đã fit, không phải một run train mới.
Family residual chỉ dự đoán runtime tại cấu hình cố định; joint
approximation/runtime target gốc vẫn chưa có score tương thích.

Các CI simulator dùng 10.000 paired hash-bootstrap replicates, là khoảng
pointwise exploratory trên các OOF predictions đã fit. Không có điều chỉnh
28 phép so sánh hoặc bootstrap retraining; không trình bày như kiểm định
confirmatory thắng mọi model.

## Bố cục deck 10 slide — dữ liệu trước, kết quả sau

1. **Câu hỏi và phạm vi:** dự đoán runtime trước khi scheduler xếp lịch cho
   hai domain; `PARTIAL`, không live-QPU và không có global leaderboard trộn
   clock.
2. **Dựng unified real-QPU ledger:** Ma–Li 340 + Qonductor 4.482 + QPack MCP
   3.945 = 8.767 observed one-circuit/service observations. Hiện label
   boundary, đơn vị, shots và identity nguồn.
3. **Bằng chứng mạch và tái dựng:** Ma–Li exact logical QASM; Qonductor exact
   submitted physical QASM; QPack sáu cấu trúc reconstruction-qualified. Góc
   và routing QPack không được phục hồi; mạch tái dựng không thay nhãn runtime.
4. **Simulator panel và điều kiện đo:** 204 members / 191 exact hashes / 22
   families, q2–q16; core 162 và frontier 42. Nêu local context, first/warm,
   quality MPS và terminal statuses; không vẽ như lưới family×width hoàn chỉnh.
5. **Protocol so sánh và method matrix:** frozen QASM/workflow groups,
   outer-train-only fitting và failure ở coverage denominator. Liệt kê sáu
   họ QPU và sáu họ simulator theo fidelity, output clock, status; 24 variants
   QPU không phải 24 paper.
6. **Real-QPU learned:** graph V3-large so polynomial cùng shared OOF rows,
   tách theo nguồn nhưng không phải ba model train riêng; gọi đúng là unified
   adaptations với representation lifecycle hỗn hợp.
7. **Real-QPU analytical:** Qiskit/QCRE scheduled proxies, Scholten nominal
   throughput và Hyb-HANAS effective-cost ở bảng diagnostic riêng. Hiện đồng
   thời evaluation-target clock và method-output clock.
8. **Simulator predictor results:** Hai bảng riêng: Aer source/hybrid/transpiled
   GNN và classical baselines trên 150 hashes; fixed-MPS graph/Ridge/residual
   trên 144 finite và companion 142 quality-pass. Nêu paired intervals và tail.
9. **Native diagnostics và methods chưa có score:** cuTensorNet estimate/actual
   cùng plan; Pasqal analog companion; Maestro pilot gate failed. Azizov local
   adaptation đã test nhưng chưa reproduce paper-wide scope; Family-Aware
   runtime-only adaptation đã test nhưng joint target gốc chưa có score.
10. **Kết luận và claims có giới hạn:** các finding được hỗ trợ, bằng chứng
    giới hạn chúng, và execution gates bắt buộc trước khi promote method còn
    thiếu.

Appendix: method cards/fidelity; actual representation; seeds/splits;
hardware/software contexts; clock boundaries; source-native unstable result;
quality/OOM/timeout; reproduction commands; public-release rights gates.

## Admit rõ những gì đã tái dựng

- Ma–Li logical QASM là exact artifacts; Qonductor exact submitted physical
  QASM không tự biến thành exact original logical circuit. 230 archive-resolved
  logical recipes là tầng riêng, không gọi byte-exact original.
- QPack thiếu original optimizer angles/submitted routing. Learned replay là
  reconstruction-qualified; analytical replay dùng representative rz(0.3)/rx(0.2).
  Labels quan sát không bị sửa thành runtime của replay.
- Learned Ma–Li/QPack dùng logical representation thực tế, khác compiled-input
  promise của S67. Nêu rõ trên method card/appendix, không tô thành faithful.
- FakeBackend nominal có thể khác ngày đo; không gọi job-day snapshot.
  T1/T2/gate duration không được thay CLOPS_v; CLOPS_h không được chuyển đổi.
- Ma–Li hybrid adaptations có DAG branch và global MLP branch, không phải
  pure graph model. Source-native 340 và Qonductor source-native 4.482 không
  được đưa thành unified 8.767 evaluations; unstable fold phải giữ nguyên.
- Family-aware local joint task, Azizov paper-wide scope, original Maestro
  Composer predictor và digital-Pasqal comparison chưa có đủ evaluated evidence.
  Azizov common-core GNN và fixed-MPS runtime-only residual adaptations đã
  có five-fold OOF evidence và phải xuất hiện trong bảng simulator hiện tại.
- Không gọi independent validation PASS là đầy đủ replication/public release.

## Chỉ dẫn cho agent dựng slide

Theo [handoff hiện tại](scientific_review.md#routine-agent-work-now-unlocked),
C1 dùng QPU pack và E6 để chuẩn bị hai bảng kết quả. C2/C4 có thể dựng figures
và deck song song sau khi bảng được kiểm tra; tối đa bốn figure hữu ích.
Giữ units, row set, clock, caption và nguồn ngay trong notes. Không chọn seed
tốt nhất hoặc tự thêm lời giải thích causal. Xuất PPTX/PDF và render QA;
final review kiểm tra deck thực tế sau khi dựng.
