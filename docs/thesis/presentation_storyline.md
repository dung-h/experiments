> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](README.md).

# Presentation storyline — 18 slides

Nội dung để người làm slide sử dụng; không tạo/ghi Canva. Chỉ copy numbers từ canonical CSV, giữ units và denominator cùng slide.

## 1. Problem

Thông điệp: Cần runtime estimates cho planning; scheduler benefit là motivation, chưa measured outcome.

Nội dung: [research_story.md](research_story.md). Evidence: [closure](../../results/research_questions/closure/README.md), SHA-256 `8301f9f97dc24a16b7d0c5f5081e7e7a11070631acab693404856629352167b7`.

## 2. Runtime semantics

Thông điệp: Observed QPU service, schedule, simulator sample/state/contraction là clocks khác nhau.

Nội dung: [rq5_fidelity_applicability.md](rq5_fidelity_applicability.md). Evidence: [registry](../../results/interpretation/thesis/estimator_fidelity_registry.md), SHA-256 `dd2589afa5117fc734d3773c6718ed2cb0fd34e8591523adae18a647e53d2a67`.

## 3. Research gap

Thông điệp: Không có fair universal ranking khi clock/input/quality và source support chưa xác định.

Nội dung: [research_story.md](research_story.md). Evidence: [claims](../../results/research_questions/closure/final_claim_matrix.csv), SHA-256 `a1badb4d2c8ecee8cec53cad4a0a2d91022082baec68c4ce2009ddbe23801d7d`.

## 4. Benchmark design

Thông điệp: Same held-out rows per method/context; grouped partitions; train-only transforms/tuning/calibration.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: contract (historical source; not bundled), SHA-256 `7903e008bfd53fd1dbcac6ffa8a983b4fc919dfa4beedee72c76253319aab4ff`.

## 5. Five RQs

Thông điệp: Accuracy, generalization, failures, accuracy–cost, fidelity; preserve closure numbering.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: [rq](../../results/research_questions/closure/final_rq_status.csv), SHA-256 `2e3d3c2dba6ef5b30f596df0aef215564aa12f0f515b9d99f9fa6f6dab3475e3`.

## 6. Datasets and contexts

Thông điệp: 4515=340+230+3945,165 groups; simulator contexts riêng và qualified survival support.

Nội dung: [research_story.md](research_story.md). Evidence: current (historical source; not bundled), SHA-256 `8180db670677e73442fb9535a22f2d3c71e2e6055e57b4728835393cb0b28b1c`.

## 7. Estimator taxonomy

Thông điệp: Native component vs architecture/core adaptation vs proxy/calibrated bridge/diagnostic.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: [taxonomy](../../results/interpretation/thesis/tables/estimator_identity.csv), SHA-256 `0c5123a41d44c2a6a1451d8be29bdd7a6d87a87480e00a1d44fe3088ec9c3504`.

## 8. Track A accuracy

Thông điệp: Graph pooled MAE 0.8341s; MLP 0.8934; selector 0.8867. Competitive paired CIs cross zero.

Nội dung: [rq1_accuracy.md](rq1_accuracy.md). Evidence: [qpu](../../results/analysis/qpu_metrics.csv), SHA-256 `958552fa0d376e7c6609cfd85f51ce3d147fd74ffb239a3054db9f432659659f`.

## 9. Source/group composition

Thông điệp: Source-balanced selector point better; equal-group MLP/Graph ordering reverses.

Nội dung: [rq1_accuracy.md](rq1_accuracy.md). Evidence: [qpu](../../results/analysis/qpu_metrics.csv), SHA-256 `958552fa0d376e7c6609cfd85f51ce3d147fd74ffb239a3054db9f432659659f`.

## 10. Generalization and failures

Thông điệp: 21-case completion; QNN/Random catastrophic errors; Kyoto collision ≠ ID leakage.

Nội dung: [rq2_generalization.md](rq2_generalization.md). Evidence: [generalization](../../results/interpretation/thesis/tables/generalization_robustness.csv), SHA-256 `59bba982441b242f355960a1f49f8df6db91b88909619416a7a511b468a9c9d4`.

## 11. Track B context results

Thông điệp: Aer compiled information, Dense strong classical comparator, fixed MPS signal; no cross-engine winner.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: [sim](../../results/interpretation/thesis/tables/track_b_findings.csv), SHA-256 `6e968c8bdf53602636d740214a65aca16278f31cb42db792f45e0f74cdec8d1c`.

## 12. Quality safety

Thông điệp: Each MPS c44 selector 64/140 unsafe choices;10 unassessed; runtime success insufficient.

Nội dung: [rq3_failures.md](rq3_failures.md). Evidence: [mps](../../results/analysis/mps_selection_safety.csv), SHA-256 `d564edf1557fbb3400bf192c6805639e9b06ee70ddc7aa76639c606496772f47`.

## 13. Why full-B2

Thông điệp: Raw parse/features/graph/IPC/inference stages; accumulated preparation fractions.

Nội dung: [rq4_accuracy_cost.md](rq4_accuracy_cost.md). Evidence: p4 (historical source; not bundled), SHA-256 `786bf48bce939d8d002d01313004e7309e7abd85515b874a92c09ccfdc71ca1a`.

## 14. QPU accuracy–cost

Thông điệp: Exact 45: accuracy advantage unresolved; warm B2 Graph≈2.22× MLP.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: p4 (historical source; not bundled), SHA-256 `786bf48bce939d8d002d01313004e7309e7abd85515b874a92c09ccfdc71ca1a`.

## 15. Dense accuracy–cost

Thông điệp: Exact 30 each precision: MLP lower MAE/B2, conditional paired CIs; full 150 neural R²≈0.14.

Nội dung: [canonical_tables.md](canonical_tables.md). Evidence: p5 (historical source; not bundled), SHA-256 `89e0ebaeb4b2111a52bba39bfd15aa618bb7243d4a3d017cc51476909eed4d26`.

## 16. Replay and fidelity

Thông điệp: 51 raw support-refined; operation order/RCCX environment semantics;210/210 exact inputs. QCEC no historical recovery.

Nội dung: [reproducibility_findings.md](reproducibility_findings.md). Evidence: [replay](../../results/reproducibility/preprocessing/validation_report.md), SHA-256 `e21e8a0d8aa00de81f580e19eca0bf6b79658f699cc626d2b0846c5cb97b16b6`.

## 17. Contributions and limits

Thông điệp: Five bounded contributions; portability/cold/classical cost/frontier and release gates remain distinct.

Nội dung: [thesis_contributions.md](thesis_contributions.md). Evidence: [contributions](../../results/research_questions/closure/thesis_contributions.md), SHA-256 `5b543d888d96025fdf0a23272ebeb97ddaf62cb8af825d2ae5020803be32e7dd`.

## 18. Conclusion

Thông điệp: Semantics/support/baselines/cost determine estimator applicability. Scientifically sufficient with material limitations.

Nội dung: [thesis_ready_conclusion.md](thesis_ready_conclusion.md). Evidence: [verdict](../../results/research_questions/closure/experiment_completion_decision.md), SHA-256 `7ae3f01cf62fea2662ab6103b259151f881e9def729944f3217f753f771fe126`.

## Những conclusions cũ phải thay

Không còn current 10/21 hoặc 12/21 generalization, cũng không primary 8767/7350 cohort. Dùng 4515 và 21/21 grid với limits. Không dùng R² của AM1 selected 30 cho full 150. B1-only latency chưa full estimation cost. Không gọi 41 là width của mọi simulator adaptation, originalScholten reproduced, Graph universal winner hay exact historical logical recovery. Old deck historical findings không bị sửa ở bước này.
