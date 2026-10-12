> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Provenance resolution

**Không còn unresolved provenance item trong 188 declarations được giao làm suy yếu primary scientific claim.** Cả 6 đúng README payload và 182 symbolic declarations đều được đối chiếu thành công. Phần author Ma–Li real tensor vẫn thiếu historical provenance; giới hạn đó đã được công bố và không được dùng để gọi local adaptation là faithful reproduction.

## Erratum: six README checks

Checker cũ `artifacts/final_exhaustive_validation/v1/build_final_exhaustive_validation.py:resolve_pin` chọn `ROOT / p` trước `manifest.parent / p`. Vì root `README.md` có thật, nó băm nhầm file cho sáu package-local payload/report entries. Producing/validating code dùng `OUT / name`, `P / name` hoặc `SUMMARY / name`; tất cả sáu README đúng namespace hiện vẫn khớp recorded hash. Không phát hiện historical target README đã evolve, mất snapshot hoặc scientific-data drift.

Category theo danh sách yêu cầu: `MANIFEST DESIGN ISSUE`, subtype **validator namespace-resolution false positive**. Convention tương đối của original manifest được xác định bằng producing code; không đề xuất sửa original manifest. Table giữ cả erroneous prior path/hash và correct path/hash, pin intent, content-preservation evidence.

Taxonomy được yêu cầu không có `VALIDATOR_RESOLUTION_ERROR`; category này mô tả vấn đề diễn giải namespace của checker, không kết luận original manifest bị sai thiết kế hoặc mất content.

| mismatch_id | historical_manifest | current_file_path | current_matches_recorded | classification | scientific_evidence_affected |
|---|---|---|---|---|---|
| H01 | artifacts/am1_feasibility/v1/provenance_manifest.json | artifacts/am1_feasibility/v1/README.md | True | MANIFEST DESIGN ISSUE | False |
| H02 | artifacts/am1_overhead_pilot/v1/provenance_manifest.json | artifacts/am1_overhead_pilot/v1/README.md | True | MANIFEST DESIGN ISSUE | False |
| H03 | artifacts/am1_phase2/v1/provenance_manifest.json | artifacts/am1_phase2/v1/README.md | True | MANIFEST DESIGN ISSUE | False |
| H04 | artifacts/am1_postpilot_audit/v1/provenance_manifest.json | artifacts/am1_postpilot_audit/v1/README.md | True | MANIFEST DESIGN ISSUE | False |
| H05 | artifacts/real_qpu/qualified_logical_panel/summary/final_validation.json | artifacts/real_qpu/qualified_logical_panel/summary/README.md | True | MANIFEST DESIGN ISSUE | False |
| H06 | artifacts/research_question_closure/v1/provenance_manifest.json | artifacts/research_question_closure/v1/README.md | True | MANIFEST DESIGN ISSUE | False |


## Symbolic declarations

182 declaration rows tương ứng 23 tên symbolic. 180 là SHA-256 bytes, hai `qcre_revision` là Git commit SHA-1 (40 ký tự), không phải SHA-256 file. Twelve Osaka/Quito neural cells lặp lại cùng 12 named code/data pins; repetition là receipt lineage, không independent evidence. Base qualified pins, source analytical receipts và generated manifests có mappings khác được ghi riêng.

Mapping được đọc từ consumer/producer tường minh, không suy ra bằng filename similarity hoặc hash search đơn lẻ. Sau khi context xác định path, bytes được băm lại; QCRE commit kiểm bằng Git HEAD và cùng-commit `estimate.py` blob, đồng thời blob SHA-256 khớp code-file pin. Supplementary path-qualified references từ inventory cũ được giữ trong provenance_chain khi có.

Classification counts: `{'RESOLVED TO AUTHORITATIVE FILE': 118, 'RESOLVED TO IMMUTABLE GENERATED PAYLOAD': 64}`. Mỗi row đã giải có `candidate_count_after_context=1`, matching identity và recorded provenance chain. Không còn ambiguous/no-path/stale declaration. Chi tiết tất cả 182 rows ở [CSV](pathless_hash_resolution.csv).

## Scientific impact

Dataset/splits/targets/predictions/checkpoints/transforms/timing/metrics không được sửa để giải provenance. Named pins chủ yếu nhận diện contract, split/support inputs, original helper/runner code và analytical cached results. Việc kiểm tra chính xác chúng bảo vệ lineage đã dùng; không tạo thêm accuracy, generalization hoặc historical-instance proof. Six README issue chỉ ảnh hưởng kết luận của generic documentation checker trước. Các 748.588 saved metric/timing/alignment checks của gói cũ và current component-validator PASS giữ giá trị trong scoped evidence của chúng.
