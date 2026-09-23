# Ma–Li flow on transpiled native DAGs

English reading path: [`docs/FINDINGS_CLUSTERS.md`](../../../../docs/FINDINGS_CLUSTERS.md) (Cluster 1, Ma–Li × Azizov). Ignore `qpu_scratch_pilot/`.

**Finding recorded:** 2026-09-22.

Experiment này giữ estimator hai tầng của Ma–Li (simulator `time_taken` → Osaka/Kyoto `result.time_taken`), nhưng thay circuit object: DAG được dựng từ `transpile(logical QASM, target backend)`, không từ source QASM. Node gắn physical qubit sau layout và T1/T2 của qubit vật lý. GNN đọc bản coarsen tối đa 256 bin theo thứ tự topological của native DAG, kèm duration/criticality và cạnh phụ thuộc. Đây không phải logical DAG trong paper Ma–Li, không phải GNN đã publish của Azizov, và không phải Ridge trên 5–7 scalar compiled features.

Eval đầy đủ chạy xong 2026-09-22 22:17 (EXIT 0, ~15 phút, CUDA). Pilot `qpu_scratch_pilot/` giữ lại như audit decode lỗi (`softplus` thay vì `expm1`); không dùng số của pilot.

## Setup

- Simulator graphs: 3020/3020 FakeWashingtonV2 / FakeSherbrooke native DAG, 2804 NPZ unique.
- QPU graphs: 340/340 FakeOsaka / FakeKyoto native DAG đã có từ `mali_physical_dag_v1` (300 unique).
- Label: public `washington/sherbrooke_time_taken.csv` và 340 Osaka/Kyoto `result.time_taken`. Hai đồng hồ không gộp.
- Split QPU: `group_fold` từ `mali_deep_protocol_v1`.
- Snapshot hiện tại, không phải calibration ngày job lịch sử. `dt` Washington/Sherbrooke `2.22e-10` s; Osaka/Kyoto `5e-10` s.
- Transfer trong script này là analogue `incl`: mô hình simulator được train trên toàn bộ 3020 hàng sim, gồm cả QASM xuất hiện ở QPU test, rồi affine / fine-tune trên fold QPU. Chưa có `excl` và family-OOD trong run này.

## Kết quả

| Model | n | log-R² | MAE (s) | MedAE (s) | R² | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| QPU scratch, coarsened native DAG | 340 | 0.9047 | 0.514 | 0.418 | 0.8735 | 0.904 |
| Simulator in-domain, coarsened native DAG | 3020 | 0.7787 | 0.269 | 0.072 | 0.4134 | 0.921 |
| Sim DAG → affine QPU | 340 | -2.6289 | 17.830 | 0.967 | -13162 | 0.763 |
| Sim DAG → fine-tune QPU | 340 | 0.8750 | 0.557 | 0.395 | 0.7316 | 0.900 |
| Affine, bỏ 2 hàng QWalk | 338 | 0.3698 | 1.172 | 0.964 | 0.3629 | — |
| Fine-tune, bỏ 2 hàng QWalk | 338 | 0.8995 | 0.505 | 0.390 | 0.8690 | — |
| QPU scratch, bỏ 2 hàng QWalk | 338 | 0.9076 | 0.501 | 0.418 | 0.8812 | — |

So với Ridge compiled-feature cùng ngày (`mali_azizov_compiled_transfer_v1`):

| So sánh | Ridge | Coarsened native DAG |
|---|---|---|
| Sim in-domain log-R² | T0 0.27 / T1 0.31 / T2 0.43 | **0.78** |
| QPU scratch log-R² | T0 0.72 / T1 0.86 / T2 0.87 | **0.90** |
| Sim → QPU affine/log-map `incl` | T2 0.35 | affine gãy; fine-tune **0.88** |

QPU scratch: median pred 8.78 s so với median label 8.64 s; không có pred = 0. Osaka log-R² 0.915 (n=192), Kyoto 0.889 (n=148).

## Đọc số

Mapped representation có tín hiệu. Trên QPU, coarsened DAG chỉ nhỉnh Ridge T1/T2 (0.90 so với 0.86–0.87): các scalar compiled đã gần ceiling. Trên simulator clock, bước nhảy lớn hơn (0.78 so với 0.43) vì `time_taken` Aer có đuôi nặng mà 5–7 số không bắt hết.

Fine-tune từ sim DAG gần scratch (0.875 so với 0.905; 0.900 so với 0.908 nếu bỏ QWalk). Nghĩa là trọng số simulator làm khởi tạo được, nhưng phần lớn khả năng fit QPU vẫn đến từ việc chỉnh lại trên 340 nhãn Osaka/Kyoto. Đây chưa phải bằng chứng frozen transfer.

Affine gãy vì hai hàng `qwalk-noancilla_indep_qiskit_9`: pred 3196 s (Osaka) và 2497 s (Kyoto) trong khi label ~13.5 s. MedAE affine vẫn 0.97 s — phần lớn mẫu không nổ — nhưng RMSE/R² bị hai điểm đó kéo. Cùng mạch đó trên simulator label 42–73 s, pred sim ~9–22 s. Affine trên giây thô không chịu được lệch thang sim (median 0.83 s, max 73 s) so với QPU (median 8.64 s, max 13.7 s). Fine-tune kéo QWalk xuống 24 s / 22 s, vẫn overshoot nhưng không còn 3000 s.

QWalk n=2 trên QPU nên R² theo family không có nghĩa. Family này vẫn là đuôi negative-transfer đã thấy ở pretrained Ma–Li.

Simulator in-domain linear R² chỉ 0.41 dù log-R² 0.78 vì đuôi: `qwalk-noancilla_9` Sherbrooke 73.2 s pred 21.6 s; `dj_16` Washington 47.6 s pred 2.8 s; một số width-30 (`ae`, `qft`, `qpeinexact`) bị under-predict mạnh. Spearman 0.92 cho thấy thứ tự đúng hơn scale ở đuôi.

## Không claim

- Không phải reproduction bit-for-bit Graph Transformer của Ma–Li trên logical DAG.
- Không phải GNN / `T_exec` đã publish của Azizov.
- Chưa có `excl` (cấm đúng QASM test khỏi sim pretraining) và chưa có family-OOD trong script này.
- Snapshot fake hiện tại không phải calibration ngày job.
- Chưa có universal QPU runtime estimator.

## Artifact

- `summary.json`, `predictions.csv`, `eval.log`
- `ws_graph_records.csv`, `PROTOCOL.md`
- coarsened cache: `coarsened/qpu` (300), `coarsened/sim` (2804)
- `qpu_scratch_pilot/` — decode bug, không dùng
