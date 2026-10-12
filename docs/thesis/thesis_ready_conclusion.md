> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# Thesis-ready conclusion

Nghiên cứu xây dựng và phân tích một benchmark cho quantum runtime estimators trên hai tracks: circuit-level observed execution/service time của real QPU và runtime của simulator dưới execution context xác định. Đóng góp nằm ở comparison protocol có runtime semantics, input qualification, shared-row evaluation, baseline strength, quality constraints và evidence traceability rõ ràng.

Trên 4515 qualified QPU observations, Ma–Li-style Graph có pooled MAE thấp hơn global MLP và classical selector trong point estimate. Paired uncertainty không xác lập ưu thế trước hai comparators cạnh tranh, và source/group weighting thay đổi ranking. Full 21-case generalization grid ghi nhận severe QNN/Random overprediction và context-blind MLP collisions trên một số backend holdouts. Những negative findings xác định ranh giới của các pipelines đã đo, không chứng minh universal unseen-hardware transfer.

Simulator results phụ thuộc target/API, noise, precision, engine và approximation. Compiled Aer information, Dense classical controls và fixed-MPS Graph có predictive signal khác nhau. Frozen MPS quality selections có failures đáng kể. Native TN estimates và analog MPS companions giữ scope riêng. Không có một simulator winner được suy ra từ các heterogeneous contexts.

AM1 warm full-B2 cho thấy parsing/preparation/bridge là substantial cost của việc tạo prediction. Trên 45 QPU observations, Graph có lower point error nhưng accuracy interval chứa zero với B2 khoảng 2.22× MLP. Trong 30-circuit Dense FP32 và FP64 subsets, MLP có lower MAE và lower B2 với conditional paired evidence. Đây là bounded accuracy–cost comparison, không operational break-even hoặc all-method deployment frontier.

Reproduction fidelity cần phân biệt semantic circuit equivalence, representation equivalence và exact frozen-model inputs. Support refinement, graph ordering và RCCX library compatibility có thể thay consumed tensors dù circuit semantics tương thích.210/210 Phase 3 inputs được phục hồi theo frozen evidence; điều này không chứng minh exact historical logical recovery hoặc original author real tensor provenance.

Final RQ1/RQ3/RQ5 được answered with qualifications, RQ4 answered within measured scope và RQ2 partially answered nhưng non-blocking. Verdict là **SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS**. No mandatory additional experiment remains. Experimental phase closed; thesis synthesis, scoped presentation và independent rights/release validation là bước bàn giao, không tiếp tục experiment queue.


Nguồn: [rq](../../results/research_questions/closure/final_rq_status.csv); [claims](../../results/research_questions/closure/final_claim_matrix.csv); [verdict](../../results/research_questions/closure/experiment_completion_decision.md); p4 (historical source; not bundled); p5 (historical source; not bundled); [replay](../../results/reproducibility/preprocessing/validation_report.md).
