> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Kiểm tra gói diễn giải khoa học

## Kết luận

**READY WITH LIMITATIONS.** Gói tài liệu diễn giải Final Scientific Freeze v2
từ evidence đã lưu. Không có experiment, inference, training, simulator timing
hay QCEC mới. Không đổi cohort, splits, feature transforms, protocols,
predictions, checkpoints, producing code hoặc historical reports.

Đây là kiểm tra reporting và saved evidence, không phải clean-checkout
reproduction, independent QPU retraining hoặc quyết định public release.

## Kết quả kiểm tra

| Kiểm tra | Kết quả | Phạm vi |
| --- | --- | --- |
| Replay gói thesis | PASS: 25 payload files, 66 direct source pins | Exact byte comparison với deterministic reporting builder |
| Focused tests | PASS: 33 tests, 6,21 s | Thesis interpretation, addendum, final reporting và full-grid consolidation |
| Addendum source validation | PASS: 1.873 source pins, 27 payload files, 16 derived tables | Chạy lại validator, không lấy nguyên PASS của snapshot cũ |
| Feature provenance | PASS: 257 transforms, 1.525 static upstream blobs | 51 raw positions; simulator 41, main QPU 40 effective dimensions |
| RQ4 authority | PASS: 21 CPU cases, 21 neural cases, 126 cells | Partitions, saved metrics, refit/checkpoint identities và historical authorization boundaries |
| Numerical reproduction | PASS: 34 QPU rows, 35 analytical controls, 49 simulator rows/7 contexts | Không pool các target clocks hoặc execution contexts khác nhau |
| Paired evidence | PASS: 21 Graph–MLP observed deltas | Delta orientation, shared rows, structural support và uncertainty eligibility |
| Historical reconciliation | PASS: 10-case/12-case snapshots, 0 metric mismatches | Historical reports bất biến; current authority là full 21-case report |
| Markdown references | PASS: 0 broken local links | Các Markdown files của reporting package |
| Whitespace check | PASS: `git diff --check`, exit 0 | Không stage, commit, push hoặc đổi Git state |

Gói chính có 21 estimator cards, 86 method-context entries, 32 claims và
crosswalk giữ 120 source questions. Có sáu bảng Markdown/CSV theo yêu cầu,
thêm analytical controls và method-context index để truy ngược. Executive
summary có 544 whitespace-delimited words, dưới giới hạn 1.000 words.

`validation_report.md` và `validation_receipt.json` là hai tài liệu dẫn xuất
đóng gói sau payload replay. Manifest pin 25 payload files và exact sources;
receipt pin manifest và báo cáo này. Không tạo manifest tự hash chính nó.
Chạy validator lần cuối sẽ kiểm tra receipt bên cạnh payload.

## Những đối soát quan trọng

### Accuracy, comparator và source weighting

Trên 4.515 observations, Graph MAE là 0,8341333282 s. Skill so với
train-only median là 26,7195604%; so với CPU selector là 5,9237157%.
Đây là hai comparator khác nhau. Paired uncertainty chưa xác lập ưu thế rõ
trước selector hoặc MLP; không đổi một point estimate thành superiority claim.
Equal-source MAE đảo thứ tự Graph/selector so với pooled MAE. Tables và
claims giữ riêng pooled, source-balanced và group-balanced interpretations.

Analytical raw output, shot-scaled output và outer-train-calibrated service
prediction được tách. Scholten generalized proxy có 3.399 eligible rows;
eligible train-median MAE 0,8200668937 s, source-median 0,6920704185 s.
Không lấy median baseline trên toàn cohort để so với subset này. Scholten
original method vẫn NOT EVALUATED, không được đổi nhãn thành reproduction.

### Hai điểm được làm rõ, không sửa kết quả gốc

1. **cuTensorNet core/frontier:** saved table
   `artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv`
   ghi core q2–q9 median ratio 3,7340122316, frontier q10–q16 4,1324156049.
   Narrative addendum trước gắn 4,1324 với core. Gói thesis dùng đúng nhãn
   từ table; giữ nguyên narrative và artifacts lịch sử. Hai medians không
   được gọi là overall median của tập kết hợp.
2. **Topology control lịch sử:** producing code
   `experiments/evaluate_mali_topology_controls.py` lấy arithmetic mean
   predictions trên mỗi observation qua ba seeds rồi mới tính metrics.
   Không phải mean của ba seed MAEs; cũng không phải median-of-three đang
   dùng trong current RQ4. Claim matrix giữ đúng aggregation và bootstrap
   riêng của control cũ. Evidence không xác lập isolated topology gain.

Không phát hiện mâu thuẫn làm thay đổi frozen primary numerical results.
Đính chính ở đây chỉ thuộc reporting; không regenerate original results.

### Generalization và negative findings

21 cases có 6.684 test assignments, nhưng chỉ 4.515 unique observation IDs.
Không overlap train/test observation IDs trong từng partition đã khai báo;
overlap giữa các holdout questions khác nhau không được coi là independent
evidence rồi pool thành một score. Family/backend/size summaries tách riêng.
Median-of-three là primary estimator đã khóa, không chọn aggregation hoặc
seed theo test performance.

Kyoto 148/148 MLP inputs collision không phải observation-ID leakage.
Graph complete representations không có cùng collision đó. Đây là
known-circuit backend-context transfer, không chứng minh hardware adaptation.
QNN và Random là held-out circuit families, không phải tên baseline models.
Cả Graph và MLP overpredict nghiêm trọng; negative R² và absolute errors
được giữ là supported failure findings, không đổi thành INCONCLUSIVE.

Uncertainty dùng structural-group support theo frozen rules. Những cases
support quá thấp không được cấp confidence interval chỉ để đủ bảng.
Full-grid completion không chứng minh universal generalization hoặc causal
hardware-generation effects. Historical Twolocalrandom execution incident
vẫn được giữ đúng authorization history.

### Feature schema, QCEC và runtime semantics

51 raw positions không đồng nghĩa 51 active inputs. Simulator upstream có
41 effective dimensions, parent QPU 40; RQ4 support masks fit-dependent.
257 saved transforms được replay; không fit normalization trên outer test.
Provenance của historical author real-QPU tensor vẫn UNVERIFIED. Graph/MLP
được gọi architecture/core adaptations, không exact Ma–Li reproductions.

QCEC counts giữ 32/32 prepared-state PROVED cho zero input theo declared
mappings, 32/32 measured-output matches trong tolerance, 8 full-unitary
PROVED và 24 COUNTEREXAMPLE dưới assumed mapping. 230 linked observations
không phải 230 independent proofs. Không chạy checker mới và không claim
exact historical logical recovery, same compilation history hay general
algorithmic equivalence. ZX inconclusive không được diễn giải non-equivalence.

QPU service time khác scheduled duration, reliability-adjusted cost và
nominal-throughput proxy. Simulator contexts cũng giữ riêng warm execution,
component execution, contraction và analog emulation. Quality failures không
bị loại khỏi reliability reporting chỉ vì runtime predictions có metric tốt.

## Những vấn đề chưa được kiểm chứng

- Historical author real-QPU tensor và exact feature path vẫn UNVERIFIED.
- Current Graph advantage trước strong selector/MLP chưa được paired
  uncertainty xác lập; source weighting ảnh hưởng ranking.
- Isolated topology contribution và hardware-generation causal effects
  chưa được chứng minh.
- Scholten original effective-depth/CLOPS compatibility chưa đủ evidence.
- cuTensorNet internal cost model không được reverse-engineer ở bước này;
  Pasqal fitted coefficients chưa được promote là identifiable model.
- Private QPU materials, ownership/author metadata và exact public payload
  còn là release gates. PASS reporting không cấp quyền phân phối hoặc cam
  kết public full QPU retraining reproduction.

Contract v1.1 vẫn có bốn RQ. RQ5 trong gói này là reporting dimension theo
yêu cầu, có explicit crosswalk, không thêm research question vào frozen
contract. Các dòng 10/21 trong contract là historical snapshot; không dùng
làm current execution status.

## Provenance và cách kiểm tra lại

Manifest (historical source; not bundled) pin exact source paths/hashes, reporting
code/tests và payload. Receipt (historical source; not bundled) lưu actual validator
outputs, test command/stdout và hash của báo cáo này. Các assertion reporting
dựa trên frozen evidence validators; không tuyên bố tái thực thi training.

Từ repo root, chạy read-only:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /home/server/Documents/.venv-qonductor/bin/python -B \
 benchmark_v1/scripts/build_thesis_interpretation.py
```

Không cần additional experiment để bảo vệ các bounded claims hiện tại.
Viết Results/Discussion từ claim matrix, giữ negative findings và
limitations. Future experiments cần research amendment riêng. Không đổi
frozen scope để có thêm kết quả đẹp. Không commit/push hoặc chỉnh Canva.
