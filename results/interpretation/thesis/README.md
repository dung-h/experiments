> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Diễn giải khoa học cho thesis

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
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /home/server/Documents/.venv-qonductor/bin/python -B \
 benchmark_v1/scripts/build_thesis_interpretation.py
```

Không training, inference, runtime measurement, QCEC, queue resume,
commit/push hoặc Canva edit. Release ownership/rights và exact public-payload
verification là gates riêng, không phải yêu cầu chạy thêm benchmark.
