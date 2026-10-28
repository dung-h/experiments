# Kế hoạch trình bày literature review — Quantum Runtime Estimation

Đề xuất tổ chức nội dung ngày 2026-10-28. Đây là dàn ý thuyết trình và chương tổng quan, không phải systematic review mới hay kết quả thí nghiệm. Chưa chỉnh slide. Các paper phải có citation và evidence card trước khi chốt câu chữ chi tiết.

## Thông điệp xuyên suốt

Ước lượng runtime cần xác định rõ: mạch nào, chạy ở đâu, cấu hình nào, và đồng hồ bắt đầu/kết thúc ở đâu. Sau đó mới đánh giá phương pháp dự đoán có chính xác, tổng quát và đủ nhanh để dùng hay không.

Người nghe cần trả lời được ba câu sau phần tổng quan:

1. Các hướng đã có dùng thông tin gì để dự đoán đại lượng thời gian nào?
2. Bằng chứng hiện có kiểm tra loại workload mới nào và còn bỏ ngỏ điều gì?
3. Vì sao benchmark của luận văn cần ghép đúng target, test population, input information và chi phí xử lý?

## Flow đề xuất: 9 slide, khoảng 10–12 phút

| Slide | Tiêu đề và câu hỏi | Hình/bảng nên dùng | Câu cần người nghe nhớ |
|---|---|---|---|
| 1 | Bài toán runtime estimation: cần biết gì trước khi dự đoán? | Circuit + execution context → estimator → một runtime có định nghĩa | Target và thông tin có sẵn lúc dự đoán là một phần của bài toán. |
| 2 | Runtime đang được đo bằng đồng hồ nào? | Timeline gồm client preparation, submission, queue, backend processing, execution, result return; mỗi paper đánh dấu phần đã chứng minh, UNKNOWN để nét đứt | Cùng gọi runtime chưa đủ để coi các nhãn là cùng boundary. |
| 3 | Bản đồ các hướng tiếp cận | Bảng analytical / tabular ML / graph-based ML / simulator-specific, với cột input và target | Cách chia nhóm xuất phát từ thông tin và mục tiêu của phương pháp. |
| 4 | Analytical: từ circuit duration đến observed runtime | Một worked example giả định: nominal circuit duration → × shots → overhead assumptions → optional learned calibration | Raw formula và calibrated predictor là hai cấp bằng chứng khác nhau. |
| 5 | Tabular ML: model thấy những thông tin gì? | Circuit statistics + shots/context được paper chứng minh → regression pipeline | Tên model chưa cho biết nó có biết hardware hay shots hay không. |
| 6 | Ma–Li: thêm biểu diễn node và graph | Logical DAG branch + global-feature MLP → regression head; ghi rõ fixed-shot evidence và transfer setup đã xác minh | Graph+MLP thay đồng thời representation và đường tính toán; so với MLP chưa cô lập connectivity. |
| 7 | Azizov: nguồn representation thay đổi ở đâu? | Ba hàng source / hybrid / transpiled, hai cột graph input và metadata; ghi execution target là noisy Aer theo review đã có | Transpiled representation là câu hỏi khác adjacency attribution; không tự biến thành real-QPU evidence. |
| 8 | Simulator runtime cần hợp đồng riêng | Contract cards: backend/method/precision/threads/shots/timing boundary; Maestro official vs component/reproduction được ghi riêng | Simulator-specific prediction và automatic backend selection cần được đánh giá riêng. |
| 9 | Khoảng trống nào dẫn tới thiết kế luận văn? | Gap → matched comparison → measurement, tối đa 4 hàng | Benchmark kiểm chứng accuracy, transfer/robustness và prediction cost trong các contract đã khai báo. |

Nếu chỉ có 6 phút: gộp slide 1+2, 3+5, giữ 4, 6, 7 và 9; chuyển simulator contracts vào appendix nếu Track B đã có phần riêng. Ithemal/GRANITE và graph-Transformer survey nên ở appendix để phần chính tập trung đúng runtime target của luận văn.

## Đặt các paper vào đúng vai trò

Đây là danh mục bố trí nội dung; không coi mỗi tên là một baseline tái lập chính thức.

| Công trình/route | Vai trò trong câu chuyện | Cần chỉ rõ trước khi trình bày | Bằng chứng local để chuẩn bị |
|---|---|---|---|
| QCRE và Qiskit duration route | Nominal operation-duration/path estimation | Unit, duration source, measurement/reset handling, circuit provenance, shots adaptation; không gán Qiskit API thành một paper riêng | `docs/audits/qpu_timing_boundary/ANALYTICAL_ROUTE_AUDIT.md`; qualification mới trong `docs/publication/RECENT_QPU_FINDINGS.md` |
| Scholten | Throughput/workload model và overhead assumptions | Công thức/assumptions gốc, loại workload và exact observed clock; đánh dấu proxy nếu local adaptation thay effective depth | `docs/research_conclusions.md`, analytical section; source implementation của frozen route |
| NoTaDS | Weighted structural score, paper/code fidelity | Paper layer score khác filtered-depth code; raw score không có đơn vị giây; calibration phải được đặt ở cột adaptation | `docs/audits/notads_hyperq_estimator_20261027/REPORT.md` |
| HyperQ | Scheduling application với duration estimate | Phân biệt paper estimator và public scheduler implementation theo revision; local replay không phải toàn bộ hệ thống | Cùng source audit; `artifacts/postfreeze_notads_hyperq_track_a_v1/REPORT.md` |
| Qonductor regression route | Classical ML và việc dùng execution context | Paper system, estimator input columns, selected local adaptation và target field phải có cột riêng | Frozen pipeline sources và `docs/research_conclusions.md` |
| Ma–Li | DAG+global learning và simulator→QPU transfer | Simulator/QPU cohorts, logical graph, feature widths, shots availability, preprocessing và checkpoint compatibility | `docs/track_a_transfer_study/CORRECTION_AUDIT.md`; Study R và A1 reports |
| Azizov | Transpilation-aware representation | Source/hybrid/transpiled view, noisy simulator target, common-set evidence, large-DAG coverage policy | `docs/final_completion/GRAPH_TRANSFORMER_LITERATURE_REVIEW.md` |
| Maestro | Simulator execution-cost estimation | Revision, available Composer implementation, calibrated host/backend/method và execution/sampling boundaries | `docs/track_b_maestro/`; QCSim component canonical report |
| Ithemal / GRANITE | Liên hệ learned cost models của classical instructions | Target gốc và input semantics khác quantum runtime; dùng làm động lực thiết kế, không làm evidence về accuracy trên QPU | Evidence card cần bổ sung citation/source trước khi dùng câu kiến trúc chi tiết |

## Evidence card cho mỗi paper

Mỗi paper chuẩn bị một trang ngắn, cùng mẫu:

- Citation, version/date, link paper và repo; code revision nếu đã đọc.
- Bài toán và deployment query: cần dự đoán cái gì, thông tin nào có sẵn trước khi chạy?
- Target: field, unit, observation unit, boundary, aggregation; UNKNOWN nếu paper/code chưa chứng minh.
- Dataset: circuit identities, execution rows, hardware/simulator, shots và cấu hình; không dùng row count thay số mạch độc lập.
- Inputs: logical/transpiled circuit, node attributes, edges, global features, hardware/context; trường nào hằng số?
- Method: learner/formula, training/calibration và preprocessing đã công bố.
- Evaluation: split unit, overlap allowed, train/validation/test, metrics/weighting, seeds, uncertainty, inference-cost boundary.
- Findings trong paper, limitations paper tự nêu và limitations mình quan sát phải có nhãn riêng.
- Local implementation fidelity: exact / adaptation / proxy / not implemented; một câu allowed claim.

Đánh dấu trạng thái evidence theo field: PAPER_VERIFIED / CODE_VERIFIED / LOCAL_ADAPTATION / UNKNOWN. Thiếu mô tả không đồng nghĩa tác giả chắc chắn không làm bước đó.

## Ma trận tổng quan để dùng trong chương viết

Cột đề xuất: work; target field/unit/boundary; QPU hoặc simulator; logical/compiled input; shots input và range; hardware/context input; estimator; calibration/training; split unit; uncertainty; prediction-cost boundary; code availability; reproduction fidelity; unresolved evidence.

Không điền một cột hardware-aware chỉ bằng YES/NO: ghi chính xác backend ID, duration lookup, calibration attributes hay coupling graph. Với shots, tách có input slot khỏi có variation trong train và có validated shot-response test.

## Cách chuyển sang benchmark của mình

Slide cuối literature review chỉ nêu các câu hỏi và phép kiểm tra; kết quả của luận văn xuất hiện ở phần Results sau đó:

| Câu hỏi | Phép so sánh cần có | Measurement |
|---|---|---|
| Estimator nào hữu ích trên observed runtime? | Learners và raw/calibrated controls trên identical held-out rows | Group-balanced MAE; pooled/source metrics và tails bổ sung |
| Simulator pretraining có giúp? | Pretrained − scratch trong cùng architecture và supervision fraction | Paired group-balanced ΔMAE + group CI |
| Graph bổ sung thông tin gì? | Graph+MLP vs MLP cho pipeline; TRUE vs NODE/SHUFFLE/REVERSE cho connectivity | Matched information/protocol, paired ΔMAE |
| Có dùng được cho query mới và có tốn thời gian không? | Group holdout / context transfer / random-row interpolation tách riêng; matched processing boundary | Coverage, support shift, tails; feature/preprocessing/inference/full-path latency |

Riêng random-row study đang chạy không có verdict. Không đưa số tạm vào literature review hay benchmark headline. Chỉ sau receipt cuối mới cập nhật Results.

## Những câu dễ bị hỏi và cách trả lời

- “Tại sao analytical sai lớn?” → Trình bày quantity dự đoán và nhãn quan sát; chưa có decomposition đo được để quy tất cả sai số cho overhead.
- “Shots có được xử lý không?” → Trình bày từng implementation; fixed-shot training không chứng minh khả năng scale shots. Kết quả thêm shots là nghiên cứu riêng của luận văn.
- “Graph tốt vì topology?” → Nêu input/capacity confounds; chỉ matched connectivity controls mới hỗ trợ attribution trong estimator.
- “Transpiled chắc chắn sát runtime hơn?” → Đây là hypothesis hợp lý cần kiểm chứng; chất lượng provenance của circuit và calibration quyết định phạm vi claim.
- “Paper A MAE thấp hơn B?” → Chỉ so trực tiếp nếu target, units, population, configuration và weighting khớp; còn lại báo cáo như kết quả riêng của từng paper.

## Trình tự thực hiện tiếp

1. Hoàn thiện evidence cards, ưu tiên Ma–Li, Qonductor, QCRE/Qiskit, Scholten, Azizov và Maestro vì trực tiếp gắn benchmark.
2. Dựng ma trận từ các cards; bất kỳ field thiếu evidence để UNKNOWN.
3. Viết phần tổng quan theo 4 nhóm phương pháp, trong mỗi nhóm dùng cấu trúc: idea → inputs/target → evaluation → limits.
4. Từ ma trận viết 3–4 research gaps có comparison/measurement tương ứng; không tuyên bố novelty bằng việc chưa tìm thấy paper.
5. Dựng 9 slide trên, mỗi slide một takeaway, citation ngay cạnh claim; chuyển implementation detail và các paper liên hệ sang appendix.
6. Review chéo target semantics và fidelity wording trước khi sửa deck. Việc hiện tại tạo plan, chưa chỉnh presentation master.
