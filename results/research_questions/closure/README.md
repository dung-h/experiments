> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Final research-question closure

Decision: **SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS**. Không còn mandatory experiment cho central thesis claim trong scope đã đo.

Đọc [provenance resolution](provenance_resolution.md), [RQ status](final_rq_status.md), [central claims](central_claims.md), [contributions](thesis_contributions.md), [gaps](final_gap_register.md) và [completion decision](experiment_completion_decision.md). [Claim matrix](final_claim_matrix.md) nối mọi kết luận với source path/hash. Các CSV lưu một declaration, RQ hoặc claim trên mỗi dòng.

Gói này bổ sung erratum cho `final_exhaustive_validation/v1`, giữ nguyên các artifact lịch sử. Six README mismatches là lỗi namespace resolution của checker trước; all 182 symbolic declarations được truy nguyên qua mapping tường minh và xác minh identity. Không thay historical manifest/hash để tạo PASS.

Working RQ1–RQ5 theo yêu cầu closure khác numbering của Research Contract v1.1: historical RQ4 generalization nằm trong working RQ2; AM1 accuracy–cost là working RQ4. Contract, frozen data/protocols/results và các historical reports giữ nguyên.

Tái tạo: `python3 -B benchmark_v1/scripts/build_final_rq_closure_v2.py`; kiểm tra byte/identity: thêm `--check`. Chỉ đọc saved evidence và read-only Git metadata. Không gọi training/inference/checker/simulator/timing workload.
