> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# Câu chuyện nghiên cứu

## Objective

Nghiên cứu so sánh các phương pháp ước lượng runtime của một circuit trên real QPU và quantum simulator, trong evaluation protocols có target semantics, input qualification và comparison population rõ ràng. Đóng góp chính là benchmark và phân tích tính áp dụng của estimator, không đề xuất một Graph model mới.

## Vì sao cần benchmark theo semantics

Một scheduled duration có thể đúng theo gate timing nhưng khác observed service time. Simulator runtime phụ thuộc engine, hardware, precision, shots, noise, approximation, API endpoint và warm/cold boundary. Một estimator tốt cho warm scalar contraction không tự dự đoán được noisy Aer execution hay CUDA-Q sample. Scheduler là motivation, chưa phải một outcome đã kiểm chứng.

## Hai tracks

Track A giữ target circuit-level observed execution/service seconds theo boundary nguồn. One-circuit job được dùng với quan hệ 1:1 đã audit, không coi mọi job/batch/workflow turnaround là circuit execution. Queue time, QPU usage và end-to-end service latency không được thay thế cho label. Cohort qualified gồm 340 Ma–Li,230 Qonductor,3945 QPack,165 conservative groups. QPack chiếm 87,4% rows nhưng chỉ 6 structural groups; reconstructed angles/layout không được gọi exact historical recovery.

Track B dùng engine/context-specific runtime labels: noisy Aer post-transpile execution, CUDA-Q Dense warm sample FP32/FP64, MPS warm state extraction và quality ladder, QCSim engine-reported component time. cuTensorNet same-plan contraction và Pasqal analog emulation là companions có endpoint riêng, không được ghép vào một leaderboard.

## Narrative theo RQ

RQ1 thiết lập accuracy trên shared held-out observations/hashes, có simple và competitive baselines. RQ2 xác định ranh giới transfer: completed negative family findings, repeated-structure backend contexts và size probes; không suy rộng mọi hardware/engine. RQ3 tổ chức failures thành prediction, representation, sampling/metric và quality-safety modes. RQ4 đo warm full pipeline prediction cost để tránh chỉ nhìn inference. RQ5 giải thích native components, adaptations và proxies, đồng thời khóa claim boundaries.

## Kết luận xuyên suốt

Graph có predictive signal trên QPU, nhưng advantage trước competitive classical pipelines nhỏ và chưa được paired uncertainty xác lập. Trong Dense exact 30, MLP có accuracy và B2 tốt hơn Graph; fixed MPS lại có strong local Graph signal. Phức tạp hơn không bảo đảm phù hợp hơn. Input/context, baseline strength, support và runtime boundary quyết định ý nghĩa của phép so sánh.

Scientific sufficiency đến từ comparison có traceability và limitations rõ, không từ universal success. Không có blocking experimental gap cho các central claims này; release metadata/rights và public reproduction vẫn là gates riêng.


Nguồn: [closure](../../results/research_questions/closure/README.md); [claims](../../results/research_questions/closure/final_claim_matrix.csv); [qpu](../../results/analysis/qpu_metrics.csv); [sim](../../results/interpretation/thesis/tables/track_b_findings.csv); [qcec](../../results/reproducibility/qcec_report.md).
