> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 Phase 4: kết luận accuracy–cost trên 45 quan sát QPU

## A. Execution verdict

**SUCCESS.** Đã hoàn tất phép đo được duyệt. Không training, refit, chạy mạch
quantum, simulator hay QCEC. Không đổi dependency, leaderboard, frozen protocol,
predictions hoặc checkpoints. Không commit/push. Không còn worker Phase 4.

## B. Gates

| Gate | Kết quả | Bằng chứng |
| --- | --- | --- |
| 0: integrity/environment | PASS, 1.487 pins | gate0.json; predeclaration.json; checkpoint_integrity.md |
| 1: raw-to-CUDA replay | Graph 45/45, MLP 45/45 | gate1.json; replay_validation.csv; replay_receipt.json |
| 2: matched full B2 | 2.700 requests, 45 × 2 × 30 | gate2.json; raw_b2_timings.csv; timing_receipt.json |
| 3: accuracy–cost | PASS, paired 44-group analysis | analysis.json; paired_observation_results.csv |

270 per-seed và 90 ensemble comparisons đều PASS với tolerance đã khóa:
`atol=1e-6 seconds, rtol=1e-5`. Cả Graph và MLP khớp model-ready input digests
độc lập từ Phase 3. Transform thực hiện đúng một lần, trước transport.

Predeclaration SHA256:
`4046a5230eb86382491d6085a870d2b4d7ac3ec1ea03ce138e376d30221c4a6e`.

Hai lỗi tooling được ghi lại, không che giấu hoặc sửa historical evidence:

- Gate 0 ban đầu đọc code pin ở acceptance snapshot đầu Phase 3 thay vì receipt
  hoàn tất. Final provenance pin khớp adapter hiện tại; fresh Phase 3 check PASS
  210/210. Xem gate0_authority_resolution.md.
- Analyzer sau đo đọc count `"341.0"` bằng `int`. Wrapper reporting mới chỉ đổi
  cách đọc metadata có giá trị nguyên trong bộ nhớ, không rounding hoặc sửa
  file. Producer/analyzer được pin vẫn nguyên trạng. Xem
  post_execution_reporting_correction.md. Không chạy lại timing.

## C. Primary results

Population: 45 IDs đã khóa, gồm 38 Ma–Li và 7 Qonductor, 44 structural groups.
Accuracy dùng original frozen median-of-three predictions. B2 là phép đo mới,
warm online, raw QASM/context đã resident đến scalar prediction ở coordinator.
Coordinator cũng là CUDA inference process. Đây không phải REST/network endpoint.

| Estimator | MAE (s) | MedAE (s) | RMSE (s) | R² | Median B2 (ms) | Median per-ID p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph | 0,724177 | 0,479326 | 0,947540 | 0,897687 | 332,557293 | 343,181681 |
| MLP | 0,812521 | 0,535865 | 1,257572 | 0,819780 | 149,513972 | 161,066421 |
| Train-only median reference | 2,864126 | 2,630567 | 3,215409 | −0,178173 | Không đo mới | Không đo mới |

Latency là median của 45 per-ID medians, không phải median gộp 1.350 repetitions.
Mỗi ID/model có 3 warmups, 30 timed repetitions, batch 1; thứ tự Graph/MLP
counterbalanced 15/15. Không xóa outlier. p95 từ 30 repetitions là nội suy,
không phải production tail bound. Raw per-repeat totals được đo trực tiếp,
không tính bằng tổng các median thành phần.

Paired structural-cluster bootstrap: 2.000 replicates, seed 20261011.

| Paired statistic | Point estimate | Conditional 95% interval |
| --- | ---: | ---: |
| Graph − MLP MAE (s) | −0,088344 | [−0,354865; 0,137886] |
| Median per-ID latency difference (ms) | 183,043322 | [150,238524; 219,504633] |
| Ratio of marginal B2 medians | 2,224256 | [2,067539; 2,408526] |

Median của 45 paired latency ratios là 2,370999, khác với ratio của hai marginal
medians. Không dùng hai thống kê này thay cho nhau. Interval mô tả uncertainty
có điều kiện trên subset đã chọn; không chứng minh representativeness.

## D. Cost attribution

| Stage: median per-ID median (ms) | Graph | MLP |
| --- | ---: | ---: |
| Parse QASM | 44,232226 | 44,357825 |
| Extract 51 globals | 90,194819 | 90,152252 |
| Prune/compact graph | 86,074837 | 0, không cần |
| Frozen transform | 5,204183 | 0,059730 |
| Array serialization | 8,076866 | 0,014186 |
| CPU payload JSON encoding | 12,338584 | 0,020388 |
| Host response JSON decode | 6,126509 | 0,020590 |
| Typed-array decode | 23,798229 | 0,036542 |
| Bridge IPC/unattributed residual | 17,665356 | 1,176434 |
| Batch/H2D | 1,592087 | 0,192853 |
| Three-seed inference + original output transform + D2H | 7,048986 | 0,578743 |
| Host ensemble median | 0,004282 | 0,004332 |

CPU spans nằm bên trong host response-wait span. Không cộng cả hai. Residual
bridge lấy hiệu duration đo ở coordinator và các duration CPU, không trừ
timestamp giữa hai process. Nó chứa pipe I/O, scheduling, outer-envelope JSON
encoding và công việc không được instrument riêng; **không phải pure IPC latency**.

Shares dưới đây dùng tổng exclusive stage durations chia tổng independently
timed B2 của đúng 1.350 requests/model, không dùng ratio of medians:

| Exclusive category | Graph | MLP |
| --- | ---: | ---: |
| CPU preprocessing | 76,11% | 97,25% |
| Serialization | 6,67% | 1,24% |
| Bridge residual | 5,18% | 0,79% |
| Host deserialization | 9,19% | 0,04% |
| Tensor construction/H2D | 0,53% | 0,19% |
| Inference/output transform/D2H | 2,30% | 0,44% |

Aggregation và coordinator residual chiếm phần nhỏ còn lại. Partition tổng
khớp từng repetition đến nanosecond. Shared parse/global extraction có thể
được reuse nếu triển khai hai model chung, nhưng đây chỉ là opportunity suy ra
từ measured stages. Không đo shared pipeline và không trừ phần này khỏi B2.

Loaded models và resident CPU server được phép; không cache circuit features.
30 frozen models được resident (5 folds × 2 models × 3 seeds). Startup/import,
model load, input disk retrieval, fold routing và external context acquisition
không nằm trong warm B2. Setup timing được ghi riêng, không phải cold B3 study.
MLP nhận cùng raw/context transport envelope nhưng không dùng T1/T2 graph
fields. JSON/base64 bridge overhead là đặc điểm implementation đo ở đây,
không phải lower bound chi phí triển khai estimator tối ưu.

## E. Scientific interpretation và source composition

Graph thấp hơn 0,088344 s về point MAE (khoảng 10,9%), nhưng paired interval chứa
0. Không xác nhận stable accuracy superiority. Graph full B2 đắt hơn rõ ràng
trong environment này, khoảng 2,22×, với interval không chứa 1.

Kết luận thực dụng: MLP là lựa chọn rẻ hơn trong measured deployment boundary.
Không có bằng chứng từ subset này rằng phần cost tăng của Graph luôn đổi lấy
một accuracy improvement ổn định. Không suy ra scheduling throughput, utility
score, scheduler benefit hoặc full benchmark Pareto frontier.

| Source | n | Graph MAE (s) | MLP MAE (s) | Graph B2 (ms) | MLP B2 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ma–Li | 38 | 0,630009 | 0,729611 | 357,897283 | 153,866288 |
| Qonductor | 7 | 1,235373 | 1,262606 | 9,622120 | 3,829921 |

Accuracy intervals cũng chứa 0 ở từng source. Qonductor có R² âm cho cả hai
model (Graph −0,442284, MLP −1,069336); không diễn giải điểm MAE nhỏ hơn là
generalization hữu ích cho mọi context. Source-balanced macro MAE là 0,932691 s
(Graph) và 0,996108 s (MLP), không phải pooled MAE hoặc source-balanced latency.

Width có liên hệ mạnh với latency trong subset: Spearman Graph 0,891, MLP
0,852; riêng Ma–Li là 0,863/0,802. Pooled depth correlations là 0,315/0,383,
nhưng riêng Ma–Li chỉ −0,137/−0,025. Đây là descriptive association có source/
workload confounding, không phải chứng minh một scaling law hoặc tác động
nhân quả của depth.

Pilot B1 chỉ đo cached-input prediction, với population/environment khác.
Thứ tự cost Graph > MLP không đảo chiều; không so trực tiếp các magnitude
B1 cũ với B2 mới như matched experiment. Phase 4 đo đầy đủ CPU preparation
và bridge cost, nên inference-only interpretation không đại diện total cost.
MLP Phase 2 ~218,77 ms cũng không được ghép với Graph mới; lần này MLP được
remeasure cùng Graph và đạt ~149,51 ms trên bridge/Qiskit đã phục hồi.

## F. RQ4 decision và limitations

**CLOSE MATCHED GRAPH–MLP AM1.** Phase 4 bổ sung bằng chứng full online estimator
cost mà cached B1 và one-sided Phase 2 chưa cung cấp. Nó không hoàn tất
classical competitive frontier hoặc mọi mục tiêu scalability của RQ4.

- Chỉ 45/50 original pilot requests có raw input; 5 QPack giữ unavailable.
  45 IDs không đại diện full cohort 4.515 và không extrapolate tới QPack.
- Competitive classical fitted states vẫn thiếu; không refit để bổ sung.
- Graph/MLP là Ma–Li-style adaptations, không exact paper training reproduction.
- Nominal snapshots và reconstructed-input qualification giữ giới hạn đã khóa.
  Không claim exact historical logical recovery hoặc lịch sử compilation giống.
- Chỉ một máy và một bridge implementation; không hardware portability,
  production latency, cold-start B3 hoặc universal scaling claim.
- Không đo scheduler-level effect. Dense FP32/FP64 B2 chưa được authorize.

## G. Integrity và resource closure

Replay: 27,11 s. Matched timing execution: 649,94 s (~10 phút 50 giây), gồm
warmups, guard và setup. Hai giai đoạn tổng khoảng 11 phút 17 giây, dưới 2 giờ.

Peak pipeline RSS được sample là 1,615 GB. Worker high-water bounds:
host 1,402 GB + child 0,843 GB = 2,245 GB; hai peaks không nhất thiết đồng thời.
Sampling 50 ms có thể bỏ qua spike ngắn, nên báo cả high-water bound này.
CUDA allocated peak 180 MB, reserved peak 220 MB; không gồm desktop GPU usage.
Toàn bộ dưới RAM/VRAM cap 8 GiB. Disk vẫn hơn 52 GB, floor 3 GiB.

Instrumentation controls: median timer-pair 55 ns; empty stdio echo 8,492 µs.
Không subtract overhead. Empty echo không mô phỏng payload Graph lớn. Timer
wrappers và monitor vẫn nằm trong direct measured total; không chạy một
uninstrumented full-B2 control để suy ra overhead chính xác toàn harness.

43 focused tests PASS (11 mới, 32 tests cũ). Fresh saved-evidence validator
PASS, `git diff --check` PASS; 1.487 source pins vẫn khớp. Independent audit
đọc lại original seed predictions/targets và raw timing totals, đối chiếu
sklearn MAE/RMSE/R² và per-ID medians: PASS với atol 1e-12, rtol 0.

## H. Next action

Dừng scope AM1 matched Graph–MLP tại đây. Không tự mở amendment khác. Muốn đo
Dense B2, competitive classical estimators, optimized transport hoặc hardware
scaling cần một authorization riêng.
