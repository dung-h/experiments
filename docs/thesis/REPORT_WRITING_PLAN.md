# Kế hoạch viết báo cáo đồ án: Ước lượng thời gian thực thi ứng dụng lượng tử

Ngày lập: 2026-10-28. Đây là kế hoạch cho báo cáo viết, bao gồm literature review và benchmark. Bản dàn ý 9 slide trước đó chỉ là nháp thuyết trình; không dùng nó làm đề cương báo cáo. Số chương/mục cần điều chỉnh theo mẫu của khoa khi có mẫu; kế hoạch này chưa giả định giới hạn số trang.

## 1. Mạch lập luận của toàn báo cáo

**Nhu cầu ước lượng runtime → định nghĩa bài toán và đồng hồ đo → các phương pháp đã có → vấn đề cần kiểm chứng → thiết kế benchmark → kết quả → giải thích và giới hạn → kết luận.**

Mục tiêu chính theo nội dung giao đề tài là triển khai và so sánh các kỹ thuật trên dữ liệu/môi trường có thể kiểm chứng. Phương pháp mới là phần mở rộng/bonus; báo cáo không phụ thuộc vào việc một kiến trúc mới phải thắng. Định thời và quản lý tài nguyên là động lực ứng dụng; automatic backend selection chưa trở thành mục tiêu thực nghiệm mặc định.

Nội dung giao đề tài và diễn giải phạm vi: `docs/research_state/THESIS_BRIEF.md`. Scientific Freeze v2 và protocol riêng của từng amendment tiếp tục giữ thẩm quyền về kết quả.

## 2. Đề cương đề xuất: 8 chương

### Chương 1 — Giới thiệu

**Mục đích:** giải thích vì sao cần runtime estimation, luận văn giải quyết phần nào và đóng góp gì.

- Bối cảnh quantum workloads và nhu cầu lập kế hoạch sử dụng tài nguyên.
- Phát biểu bài toán: dự đoán một runtime đã định nghĩa từ circuit, execution context và thông tin có trước khi chạy.
- Phạm vi hai Track: archived real-QPU observations và simulator contracts được đo trong từng môi trường cụ thể.
- Câu hỏi nghiên cứu, đóng góp, phạm vi và bố cục báo cáo.
- Đóng góp viết thành sản phẩm/bằng chứng: tập benchmark có provenance; triển khai/adaptation có fidelity khai báo; đánh giá matched; transfer/robustness/cost diagnostics. Chỉ liệt kê method mới ở mức bằng chứng thực sự đạt được.

**Cần có:** một sơ đồ pipeline tổng quát và bảng phạm vi trong/ngoài đề tài. Abstract và phần đóng góp được hoàn thiện sau khi Results đã chốt.

### Chương 2 — Cơ sở lý thuyết và đặc tả bài toán

**Mục đích:** người đọc hiểu các khái niệm cần dùng để đọc phương pháp và benchmark.

- Circuit, gate, qubit, measurement, parameter, depth và shots.
- Logical circuit, transpiled/native circuit, circuit dependency DAG và QPU coupling graph; định nghĩa chúng riêng và minh họa trên cùng một mạch nhỏ.
- QPU execution lifecycle: preparation/compilation, submission, waiting, backend processing, measurement và result retrieval; thành phần không có bằng chứng để UNKNOWN.
- Simulator methods và các cấu hình làm thay đổi runtime: statevector, noisy simulation, MPS, precision, threads, chi/truncation và sampling.
- Runtime target, observation unit, repetitions và aggregation. Phân biệt thời gian workload chạy với thời gian estimator tạo prediction.
- MAE/RMSE, raw R² và log1p R²; group/source/row weighting; ý nghĩa CI và paired comparison.

**Cần có:** hình lifecycle với boundary được khai báo; bảng thuật ngữ và ví dụ MAE/R². Không diễn giải `Result.time_taken` thành thời gian xung vật lý thuần nếu chưa chứng minh.

### Chương 3 — Tổng quan nghiên cứu liên quan (literature review)

**Mục đích:** tổng hợp các hướng nghiên cứu, đánh giá evidence và dẫn tới lựa chọn benchmark.

**3.1. Phạm vi và cách khảo sát.** Ghi nguồn tìm paper, từ khóa, thời điểm khảo sát, tiêu chí đưa vào và loại ra. Phân biệt paper gốc, repository, tài liệu API và implementation local. Nếu chưa có quy trình tìm kiếm/tuyển chọn đầy đủ thì gọi đây là targeted review, không tự nhận systematic review.

**3.2. Analytical và hardware-informed estimation.** QCRE, Qiskit duration route, Scholten và các phương pháp có liên quan như NoTaDS/HyperQ. Với mỗi phương pháp: quantity gốc → thông tin cần có → công thức/giả định → evidence đánh giá → limits. Raw score, shot multiplication và learned calibration có tên riêng. Qiskit API là một route triển khai, không tự coi như một paper nghiên cứu độc lập.

**3.3. Traditional ML/regression.** Những gì model được phép biết về circuit, shots và backend/context; model, preprocessing, target và split. Đối chiếu vai trò estimator trong các hệ thống như Qonductor với local adaptation của luận văn.

**3.4. Graph-based learning và transfer.** Ma–Li: node/global representation, logical DAG, simulator/QPU targets, pretraining/fine-tuning, fixed-shot support và giới hạn tái lập. Phân biệt graph-pipeline benefit với incremental connectivity value.

**3.5. Compilation-aware representation.** Azizov và các hướng dùng transpiled circuits/metadata. Kiểm tra environment/target thực tế của paper; source/hybrid/transpiled là thay đổi representation, không tự chứng minh edge-topology attribution hoặc real-QPU generalization.

**3.6. Simulator-specific estimation.** Maestro và các hướng phù hợp từng simulator/method. Khai báo host calibration, method/configuration, official implementation availability và target boundary. Không trộn predictor với automatic backend/method selector.

**3.7. Tổng hợp và research gaps.** Dùng bảng đối chiếu để kết luận phần nào đã có evidence, phần nào thiếu matched comparison. Từ đó dẫn tới accuracy, generalization, failure, cost và fidelity questions của báo cáo.

Trong mỗi nhóm, cấu trúc đoạn viết là **vấn đề → ý tưởng chung → các khác biệt quan trọng giữa paper → evidence và giới hạn → câu hỏi còn lại**. Không biến literature review thành chuỗi tóm tắt từng abstract. Ithemal/GRANITE có thể tạo liên hệ ngắn với learned cost models; chi tiết chỉ giữ nếu giúp giải thích lựa chọn phương pháp, và cần evidence card trước khi viết kiến trúc cụ thể.

**Cần có:** bảng literature matrix gồm work/version, target/unit/boundary, QPU/simulator, circuit representation, shots/context inputs, model, data/split, metrics/uncertainty, prediction-cost scope, code/data availability và fidelity của local implementation. Mỗi substantive claim phải có citation paper/code; kết quả luận văn chuyển sang chương 5–6.

### Chương 4 — Thiết kế và triển khai benchmark

**Mục đích:** mô tả đủ để hiểu một con số được tạo ra thế nào và có thể tái lập trong phạm vi dữ liệu công bố.

**4.1. Câu hỏi và comparison map.** Dùng RQ crosswalk hiện có: RQ1 accuracy; RQ2 generalization; RQ3 error/robustness; RQ4 accuracy–cost; RQ5 fidelity/applicability. Giữ historical RQ IDs ở nguồn và ghi crosswalk khi trích dẫn, không đổi nhãn các artifact cũ.

**4.2. Dữ liệu Track A.** Provenance ba nguồn, common `Result.time_taken` lineage, conversions/aggregation, source-QASM/physical-QASM/reconstruction availability, screening và cohort 4.515 observations/165 groups. Giải thích grouping bằng ví dụ mạch có cùng chuỗi operation/operands nhưng khác góc quay. Counts của exact QASM và structural groups phải lấy từ manifest, không dùng lẫn.

**4.3. Dữ liệu Track B.** 150 core exact source identities; từng contract và observation key riêng. Joint MPS là 150 circuits × 6 chi = 900 observations. Native contraction/analog studies đặt thành companion studies nếu không cùng core/target.

**4.4. Estimator implementations.** Mỗi phương pháp có sơ đồ input→preprocess→model/formula→output. Bảng feature/input nêu rõ logical/transpiled, node/global, shots, duration lookup/backend ID/calibration/coupling inputs. Ghi exact reproduction, reconstruction, adaptation hoặc proxy theo evidence; không gộp chúng bằng tên paper.

**4.5. Training/evaluation.** Outer/inner split, unit/group, scaler/imputation/vocabulary fitting, seeds, checkpoint selection, fixed candidate budgets và test-label isolation. So sánh Graph/MLP/classical là pipeline comparison khi inputs hoặc tuning/capacity khác nhau.

**4.6. Metrics/inference.** Mỗi study có metric hierarchy và seed-reduction theo protocol gốc; báo cáo rõ weighting và raw/log target scale. Group-balanced metrics là headline cho extension nào đã prespecify như vậy; pooled frozen results giữ nguyên tên/estimand lịch sử. Paired bootstrap dùng unit của study, không independent rows khi rows cùng group.

**4.7. Chi phí estimator.** Tách feature/preprocessing, warm fitted-model inference và QASM→prediction; offline training/calibration ghi riêng nếu có. Mỗi bảng phải khai báo CPU/GPU, batch, warmup, repetitions và returned-output boundary.

**4.8. Reproducibility.** Freeze/amendment separation, source/config/input hashes, software/hardware, saved predictions, failures và release scope. Báo cáo public verification khác private-data replay.

**Cần có:** dataflow figure; cohort/provenance table; contract inventory; method/feature table; comparison/metric table; leakage receipts và environment appendix.

### Chương 5 — Kết quả và phân tích trên QPU (Track A)

**Mục đích:** trả lời câu hỏi bằng matched experiments đã hoàn tất, theo thứ tự từ benchmark chính tới kiểm tra mở rộng.

1. Dataset profile, effective input diversity và coverage trước bảng score. Đặc biệt QPack có 3.945 observations nhưng chỉ 6 raw global-feature vectors trong input MLP; không áp con số đó cho toàn bộ graph input.
2. Frozen grouped benchmark: cùng held-out rows, accuracy, coverage, uncertainty và weighting sensitivity. Báo pooled/source/group-balanced cạnh nhau để giải thích câu hỏi trọng số.
3. Analytical controls: raw → shot-scaled → outer-train calibrated trên identical eligible rows; report per-source và raw-to-observed mismatch. Không gán residual cho measured overhead chưa có decomposition.
4. Transfer: Study R như historical motivation; A1 matched full-QPU rồi reduced fitted-group supervision extension như evidence riêng. Ghi rõ hai subset draws và validation-label availability của extension; không gọi thành generic total-label learning curve.
5. Shots-input ablation: cùng folds/inputs/training trừ scalar shots; effect+CI, source/fold heterogeneity và failure diagnostics. CI crossing zero không chứng minh shots vô ích.
6. Representation/topology: graph-vs-global, source/transpiled views và matched connectivity controls theo đúng cohort từng study. Study R 340 rows không thay bằng Track-A 4.515 rows. P1 representation gate không đạt không phải bằng chứng mọi physical DAG model vô ích.
7. Generalization: family holdout, backend-context transfer, small→large size probes; test population/support/target variation và prediction direction. Nêu representation overlap khi cùng input từng có trong train.
8. Prediction cost trong đúng phạm vi v4: ready normalized CPU input → final prediction/host scalar. QPU MLP/Graph đã đo; classical QPU/full raw-QASM path còn thiếu thì ghi giới hạn.
9. Random-row source×backend-stratified study: **chờ completion/validation receipt**. Nếu đủ kết quả mới có mục interpolation/context-reuse riêng và chỉ so split effects trên cùng pipeline/protocol. Nếu chưa xong lúc chốt báo cáo thì để ongoing/future work, không dùng partial cells làm headline.

**Cần có:** main score table; weighting comparison; paired-delta forest plot; analytical ladder; transfer curve; error/support plot; generalization case table; latency table có boundary.

### Chương 6 — Kết quả và phân tích trên simulator (Track B)

**Mục đích:** đánh giá estimator trong từng execution contract đã đo.

- Bắt đầu bằng canonical contract/coverage inventory, rồi lần lượt Dense FP32/FP64, noisy Aer, QCSim CPU và MPS fixed/joint R9.
- Trong từng contract: observation key/configuration/clock → estimators → matched accuracy và coverage → cost/quality nếu có → allowed claim.
- QCSim recovered 150 result là current canonical; legacy 143 result là provenance, không tạo thêm một current leaderboard entry. Ghi Maestro-style component thay official multiengine predictor nếu fidelity chỉ tới mức đó.
- Joint-MPS báo coverage cả hash và hash×chi; chất lượng output là trục riêng. Nếu bàn selector, phải công bố quality failures/abstention và support của evaluation.
- cuTensorNet same-plan contraction và Pasqal analog studies được trình bày trong scope riêng hoặc appendix; không ép vào bảng universal simulator leaderboard.
- Cost v4 Dense dùng đủ 150 hashes mỗi precision nhưng boundary vẫn ready-input inference; không thay bằng full QASM→prediction claim.

**Cần có:** canonical inventory table, mỗi contract một score panel, quality/runtime plot khi phù hợp, cùng target/configuration note ngay cạnh metric.

### Chương 7 — Thảo luận, tính hợp lệ và giới hạn

**Mục đích:** tổng hợp ý nghĩa của các kết quả, phân biệt mô tả với giải thích đã kiểm chứng.

- Trả lời từng RQ bằng claim + direct comparison + evidence + limitation; bảng RQ→answer→artifact.
- Ranking phụ thuộc context, information availability và weighting; point winner khác established paired improvement.
- Những kết quả bất thường: neural OOD bias, MAE nhỏ cùng R² âm, shot/source confounding, calibration gain và unsafe quality selection. Nêu hypothesis khi mechanism chưa được kiểm chứng.
- Internal validity: splits, preprocessing, feature availability, hyperparameter budgets, seed variability và ensemble reduction.
- Construct validity: provider runtime field, unknown timing components, nominal snapshots, proxy definitions và reconstructed inputs.
- External validity: nguồn dữ liệu ít independent groups ở một số source, host/backend/date range, partial feature coverage và archive selection.
- Statistical/reproducibility limits: conditional bootstrap, subset draws, multiple analyses, private-input release limits và reproducible scope của public package.
- Method-development pilots: chỉ dành phần ngắn cho lessons hoặc appendix nếu không tạo primary contribution; negative results được giữ với bounded claim.

**Cần có:** claim/evidence/limitation matrix. Không chọn lời giải thích có vẻ hợp lý rồi viết như nguyên nhân đã đo được.

### Chương 8 — Kết luận và hướng phát triển

- Tóm tắt câu hỏi, thực nghiệm và câu trả lời trong đúng scope.
- Đối chiếu các yêu cầu giao đề tài với sản phẩm và bằng chứng đã tạo.
- Đóng góp chính: benchmark protocol/provenance, triển khai/adaptation, comparative evidence, failure/cost analysis.
- Hướng phát triển xuất phát từ uncertainty còn lại: new controlled data, timing metadata, validated shot-response/cross-context evidence hoặc method extension có comparison cụ thể. Không hứa một architecture mới chắc chắn khắc phục negative finding.

## 3. Giữ các lớp bằng chứng riêng trong văn bản

| Lớp evidence | Vai trò | Cách trích trong báo cáo |
|---|---|---|
| Literature gốc | Tác giả đã đề xuất và kiểm chứng gì | Paper/code citation; không dùng local adaptation score làm original published result |
| Study R | Historical Ma–Li reconstruction/topology evidence | Cohort/split/seed reduction và reconstruction limits ghi riêng |
| Frozen Track A/B | Primary benchmark evidence | Canonical manifests, exact populations/contracts, saved metrics và current Track-B index |
| Post-freeze amendments | Bounded new comparisons | Tên study đầy đủ, protocol, result/CI và completion receipt riêng |
| Diagnostics/pilots | Giải thích mô tả và hypothesis generation | Post-hoc/exploratory label; không nâng thành causal mechanism hoặc selected universal method |

## 4. Evidence index cho quá trình viết

Các đường dẫn dưới đây trỏ tới evidence local; không khẳng định mọi private input được phát hành trong public report checkout.

| Phần viết | Entry point cần đọc |
|---|---|
| Phạm vi và RQ | `docs/research_state/THESIS_BRIEF.md`; `artifacts/research_question_closure/v1/rq_framework.md`; `artifacts/final_rq_closure_v2/final_rq_status.md` |
| Current findings | `docs/research_conclusions.md`; `docs/final_completion/CLAIM_MATRIX.md`; `docs/publication/RECENT_QPU_FINDINGS.md` |
| Frozen main QPU | `artifacts/real_qpu/qualified_logical_panel/`; freeze manifest và producing protocol |
| Target lineage/corrections | `docs/audits/qpu_timing_boundary/RESULT_TIME_TAKEN_PROVENANCE_NOTE.md`; latest source/implementation audit qualification |
| Transfer | `artifacts/postfreeze_track_a_transfer_v1/FACTORIAL_100_RESULTS.md`; `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/P1_SAMPLE_EFFICIENCY_REPORT.md` |
| Shots | `artifacts/postfreeze_mali_shots_ablation_v1/FINAL_REPORT.md` và paired/completion receipts |
| Source/transpiled | `artifacts/postfreeze_track_a_transpiled_representation_v1/FINAL_REPORT.md` |
| Topology historical | `artifacts/postfreeze_dag_raw_r2_audit_v1/DAG_RAW_R2_VERIFICATION.md`; `DAG_TOPOLOGY_MECHANISM_AUDIT.md` |
| Generalization | Frozen 21-case full-grid reports và source prediction/feature-support artifacts, resolved through current evidence registry |
| Current Track B | `docs/track_b_finality/TRACK_B_CANONICALIZATION_REPORT.md`; canonical contract index referenced there |
| Cost current measured scope | `docs/final_completion/ESTIMATOR_COST_V4_COMPLETION_REPORT.md` và v4 summary/receipts |
| Random-row ongoing | `artifacts/postfreeze_track_a_random_row_benchmark_v1/PROTOCOL.md`; live process/progress và final receipt khi có |

Historical RQ/Track-B documents có thể còn số 143/840 cũ; chúng không supersede recovery/R9 canonical artifacts. Tương tự, cost completion report cũ không thay v4 trong scope ready-input inference. Luôn đối chiếu original result và correction note trước khi ghi số.

## 5. Thứ tự làm việc để viết hiệu quả

| Bước | Công việc | Sản phẩm đạt yêu cầu |
|---|---|---|
| 1 | Lập report evidence matrix: claim, chapter, source, cohort/contract, metric, uncertainty, status | Mỗi dự định đưa số/claim đều có đường dẫn evidence và chưa trộn studies |
| 2 | Viết chương 4 từ manifests/code/protocol | Người đọc xác định được unit, inputs, splits và metric của mỗi comparison |
| 3 | Lập bảng/hình chính chương 5–6 từ saved validated aggregates | Caption có population, unit, weighting, contract và uncertainty; không train lại để có số đẹp |
| 4 | Hoàn thiện evidence cards và viết chương 3 | Tổng hợp các hướng, ưu/giới hạn và research gaps với citation; không dùng benchmark result làm published claim |
| 5 | Viết chương 2 theo những khái niệm thực sự cần cho chương 3–6 | Không biến background thành giáo trình quantum/ML quá dài |
| 6 | Viết chương 7, 8 rồi hoàn thiện chương 1 và abstract | Mọi đóng góp trong mở đầu đều có kết quả hoặc sản phẩm trong thân bài |
| 7 | Review citations, metric scale/units, counts, target lineage, current canonical pointers và claims | Một người khác trace được headline tới artifact; UNKNOWN và negative evidence còn hiện rõ |

Khóa nội dung từng phần sau khi kiểm tra, tránh để một pilot/new amendment kéo theo viết lại toàn bộ story. Phần đang chạy chỉ bổ sung sau completion/validation; writing progress có thể tiếp tục ở các chương khác.

## 6. Phụ lục đề xuất

- Method/feature dictionaries, gate categories và backend snapshot registry.
- Cohort screening/provenance, structural-group definition và split/OOF integrity.
- Hyperparameter grids, model dimensions/seeds và training receipts.
- Complete per-source/backend/fold tables, tails và implementation/fidelity audits.
- Historical/recovery revision crosswalk, especially QCSim legacy→150 và MPS R9.
- Environments, commands, artifact hashes, publication boundaries và access requirements.

Thân bài giữ các phép so sánh trực tiếp trả lời RQ; phụ lục giữ chi tiết cho việc kiểm tra và tái lập. Không cần một mục riêng trong thân bài cho mọi lần chạy hoặc mọi file trong repository.
