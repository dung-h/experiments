> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 Phase 5: Dense Graph–MLP accuracy–cost

## A. Verdict

**SUCCESS.** Hoàn tất phạm vi được duyệt. Không training/refit, chạy simulator,
QPU, QCEC, tạo mạch, đổi dependencies hoặc sửa frozen results. Không commit/push.
Không còn worker Phase5. Scope kết thúc tại hai Dense contexts.

## B. Gates

| Context | Gate0 integrity | Gate1 raw-to-CUDA replay | Gate2 matched full B2 |
| --- | --- | --- | --- |
| FP32 | PASS, 30 IDs | Graph30/30, MLP30/30 | PASS, 1,800 requests |
| FP64 | PASS, 30 IDs | Graph30/30, MLP30/30 | PASS, 1,800 requests |

1,569 pins khớp trước/sau execution. Replay gồm360 seed và120 median3
comparisons, tất cả PASS. Max absolute drift7.450580596923828e-9 seconds.
Tolerance khóa trước:atol1e-6 seconds+rtol1e-5. Independent Phase3 input
digests xác minh cả dtype,shape,globals,node features và edge indices.

Predeclaration SHA256:
`83b0ce5a29f5083a4ba18d4df163455724a54aef3204e23a093d02f158150bca`.
Exact raw inputs,60 checkpoints,fold transforms và source routing ở
predeclaration.json. completed Phase3 provenance là authority cho compatibility
code; không thay pin ở historical acceptance snapshot.

## C. Matched results

Accuracy dùng frozen median-of-three predictions, không dùng output timing
để chọn kết quả. Mỗi context có30 circuits/30 QASM-hash groups,6/fold.
Hai subsets có20 hashes chung và40 unique hashes tổng cộng. Không coi60
context requests là60 independent circuits khi so chéo precision.

| Context | Estimator | MAE (ms) | RMSE (ms) | R² | Median B2 (ms) | Median per-ID p95 B2 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Graph | 1.753429 | 3.596786 | 0.678366 | 7.250092 | 7.620789 |
| FP32 | MLP | 1.523329 | 3.386699 | 0.714842 | 3.689708 | 3.971862 |
| FP64 | Graph | 0.807979 | 1.625005 | 0.925081 | 7.076096 | 7.512217 |
| FP64 | MLP | 0.618299 | 1.146406 | 0.962713 | 3.610452 | 3.892993 |

MAE/RMSE là sai số dự đoán simulator runtime, B2 là chi phí tính estimator.
Hai đại lượng có cùng đơn vị ở bảng nhưng đo clock khác nhau.
Median B2 là median của30 per-ID medians. p95 là median của30 per-ID
interpolated p95 từ30 repetitions, không phải pooled/production tail bound.
Không loại outlier.3 warmups/circuit/model,30 repeats,batch1,counterbalanced15/15.

Paired cluster bootstrap2,000 replicates,seed20261011,exact existing QASM-hash
groups trong từng context. Repetitions không là independent circuits.

| Context | Graph−MLP MAE (ms), 95% CI | Median paired B2 difference (ms), 95% CI | Ratio of marginal medians, 95% CI |
| --- | --- | --- | --- |
| FP32 | 0.230100 [0.056417,0.459004] | 3.555169 [3.275192,3.924500] | 1.964950 [1.916280,2.013973] |
| FP64 | 0.189680 [0.027737,0.413600] | 3.465644 [3.100782,3.829302] | 1.959892 [1.897088,2.010439] |

Median paired ratios1.914535/1.917204 khác ratio of marginal medians.
Không gán interval của thống kê này cho thống kê khác.

Full frozen core OOF có150 circuits/context, không phải30 timed circuits:

| Context | Estimator | n | MAE (ms) | RMSE (ms) | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | Graph | 150 | 1.715940 | 8.692055 | 0.134538 |
| FP32 | MLP | 150 | 1.577722 | 8.654185 | 0.142063 |
| FP64 | Graph | 150 | 1.730218 | 8.636725 | 0.135541 |
| FP64 | MLP | 150 | 1.590082 | 8.593155 | 0.144241 |

Không đưa subset R² cao vào claim full-context accuracy. Full scores giữ
nguyên, không được dùng làm lý do tuning/refit mới.

## D. Cost decomposition

Median per-ID median của stage, milliseconds:

| Stage | FP32 Graph | FP32 MLP | FP64 Graph | FP64 MLP |
| --- | ---: | ---: | ---: | ---: |
| Parse QASM | 2.336998 | 2.340232 | 2.292650 | 2.286821 |
| RCCX/circuit normalization | 0.044010 | 0.043731 | 0.038340 | 0.039750 |
| Seven globals | 0.244442 | 0.242391 | 0.184427 | 0.183925 |
| DAG construction | 0.141422 | 0, không cần | 0.101888 | 0, không cần |
| Frozen feature encoding/transform | 0.382961 | 0.045875 | 0.309033 | 0.044563 |
| Bridge/unattributed residual | 0.092355 | 0.074944 | 0.083305 | 0.075342 |
| Batch construction/H2D | 0.409013 | 0.250719 | 0.407306 | 0.244319 |
| Three-seed inference/inverse/D2H | 3.267004 | 0.481010 | 3.301021 | 0.491228 |
| Host median aggregation | 0.002911 | 0.002690 | 0.002880 | 0.002691 |

Không cộng stage medians để tái tạo median B2. Total được đo trực tiếp.
CPU stages nested trong host response wait. Residual là duration difference,
không subtraction timestamp hai process và không gọi là pure IPC.
Inference stage bao gồm original expm1/clamp và D2H, không pure GPU kernel time.

Shares dùng tổng exclusive durations/tổng direct B2 của900 requests/model/context:

| Category | FP32 Graph | FP32 MLP | FP64 Graph | FP64 MLP |
| --- | ---: | ---: | ---: | ---: |
| CPU preparation | 67.63% | 86.16% | 62.80% | 83.09% |
| Inference/inverse/D2H | 23.44% | 6.62% | 28.05% | 8.27% |
| Serialization | 1.74% | 0.87% | 1.60% | 0.99% |
| Host deserialization | 1.78% | 0.49% | 1.57% | 0.60% |
| Bridge residual | 1.56% | 1.14% | 1.47% | 1.32% |
| Tensor construction/H2D | 3.54% | 4.13% | 4.14% | 5.01% |

Phần còn lại là aggregation/coordinator residual. Mỗi partition khớp total
đến nanosecond. Không chạy ablation để quy toàn bộ cost difference cho graph.

## E. Within-range behavior

Active width2–9,normalized operations4–3,152,depth tới3,034,edges tới6,168.
`gates` trong metadata chính là operation nodes, gồm measurements.
Spearman latency–operation count Graph/MLP:0.994/0.985 FP32,0.986/0.964 FP64.
Depth khoảng0.960–0.972; width chỉ khoảng0.561–0.605.
Đây là associations trong selected mixed-family range. Không chứng minh
asymptotic/exponential scaling hay complexity law của simulator.

## F. Scientific interpretation

Trong hai exact30 contexts, MLP có MAE thấp hơn và cost thấp hơn Graph,
conditional intervals cùng ủng hộ chiều này. Không thấy accuracy benefit
của Graph để bù cost~1.96×. Pairwise dominance giới hạn ở hai tiêu chí và
hai subsets này, không suy thành all-method/universal winner.

Train-only median reference có MAE2.135361/1.437864 ms và R² âm; tuy nhiên
MedAE0.179651/0.171514 ms tốt hơn cả hai neural models. Phần lớn small-runtime
circuits và tail errors vì vậy không được diễn giải bằng một winner duy nhất.
Reference cost không đo lại.

B1 cached-input cũ~3.23/0.49 ms và3.21/0.49 ms bỏ qua parsing/preparation.
Full B2 xác nhận những chi phí này đáng kể. Không tính matched B2/B1 speedup
từ hai execution occasions/warmup policies khác nhau.

Descriptive B2/historical simulator-runtime ratios Graph/MLP:30.25/15.41
(FP32),34.33/17.40(FP64),và B2 lớn hơn historical label ở30/30 từng model.
Đây là **historical cross-environment simulator runtime**, không matched
execution, operational break-even,throughput benefit hoặc scheduler evidence.
Không chạy simulator mới để sửa giới hạn đó.

## G. RQ4 Track B

**ANSWERED WITHIN DENSE CONTEXT.** Đã nối frozen accuracy với full online
estimator cost dưới cùng external boundary. Không hoàn tất classical frontier,
mọi simulator context,hardware portability,cold B3 hoặc production utility.
Graph/MLP là architecture adaptations; simulator FP32/FP64 không có nghĩa
model chạy FP32/FP64: cả hai frozen estimator bundles dùng float32.

## H. Integrity and resources

55 focused tests PASS(14.36 s),fresh independent sklearn/readback audit PASS,
git diff--check PASS.1,569 source pins không đổi;60 checkpoint identities và
480 replay comparisons hợp lệ.18 required deliverables có đủ.

Replay/timing tổng57.805 s. Peak sampled aggregate RSS1.356 GB,conservative
host+child high-water bound2.030 GB(không simultaneous). CUDA allocated68.3 MB,
reserved79.7 MB. Disk còn~52.45 GB. Các giới hạn8GiB/2h/256MiB/3GiB floor
đều được giữ. Artifact payload~3.8 MB; exact bytes và hashes ở final receipt.

Timer controls57/54 ns,empty echo~8.54 µs,không subtract overhead.
Instrumentation/resource-monitor overhead nằm trong direct B2. Resident
models/processes được phép; imports/startup/model loading/input disk reads
không nằm trong warm B2. Không cache circuit features,không CPU inference
fallback,không background benchmark worker còn lại.

## I. Next action

**CLOSE DENSE GRAPH–MLP AM1.** Dừng tại phạm vi đã duyệt. Bất kỳ benchmark
context mới hoặc competitive-classical measurement nào cần amendment riêng.
Không commit/push hoặc chỉnh Canva/Notion/LaTeX trong lượt này.
