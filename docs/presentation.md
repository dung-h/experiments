# Nội dung khoa học cho báo cáo benchmark

Phạm vi: báo cáo nội bộ khoảng 10 phút, tiếng Việt. Đây là nguồn nội dung duy
nhất cho deck; không tạo một outline/version song song. Số liệu phải lấy từ
`benchmark_v1/execution/manifests/report_finalization.json` và artifact mà nó
pin. Bản manifest này khóa bằng chứng số, không khóa cách diễn giải hay bố cục
slide. Không training, đo thêm hoặc thử seed mới trong queue dựng báo cáo.

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
408 candidate attempts chưa chạy; không có accepted predictor/MAE score.
Clock là reported execution trong fresh process, không gọi warm persistent
process. Chưa xác định nguyên nhân variation; không đổ lỗi cho threading,
hardware hay thuật toán khi chưa có intervention evidence. Không suy Maestro
paper predictor thất bại trên common panel từ pilot này.

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
8. **Simulator method results:** Aer local graph/Ridge/GBR core; cuTensorNet
   estimate/actual cùng selected plan; CUDA-Q coverage và MPS quality failures.
   Phân biệt engine đã đo với predictor đã benchmark.
9. **Methods chưa có score hợp lệ:** Azizov GNN chưa có signed local run,
   Family-Aware thiếu label chung phù hợp, Maestro pilot gate failed; Pasqal là
   analog companion, không phải digital-QASM row.
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
- Family-aware local joint task, Azizov common-panel GNN, original Maestro
  Composer predictor và digital-Pasqal comparison chưa có đủ evaluated evidence.
- Không gọi independent validation PASS là đầy đủ replication/public release.

## Chỉ dẫn cho agent dựng slide

F-C1 phải sinh và validate final pack v3 trước khi chèn số cuối. F-C4 có thể
dựng tối đa bốn figure hữu ích; không cần vẽ đủ mọi finding. Giữ units, row set,
clock, caption và nguồn ngay trong notes. Dùng đúng brief này, không chọn seed
tốt nhất, không tự thêm số hoặc lời giải thích causal. F-C5 xuất PPTX/PDF và
render QA; F-S4 duyệt sau khi có bảng/package/deck, không ký sẵn.
