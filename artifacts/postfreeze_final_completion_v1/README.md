# Final completion audit namespace

This namespace contains only documentation-derived diagnostics and indexes for the final-completion campaign.  Frozen Study-R, Track-A, Track-B, and A1 artifacts are not overwritten.

- The sample-efficiency extension completed under `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/`; 360/360 cells passed validation. Its conditional low-fitted-group MLP result is described in `P1_SAMPLE_EFFICIENCY_REPORT.md`; it does not replace the frozen 100%-QPU conclusion or establish a Graph benefit.
- The existing Graph/MLP cost evidence remains under `artifacts/am1_phase4/v1/` and `artifacts/am1_phase5_dense_b2/v1/`; a complete classical timing frontier was not run because no comparable raw-input harness is present.
- OOD values are traceable to `docs/generalization_evaluation.md`, `artifacts/final_research/rq4_report.md`, and `artifacts/real_qpu/generalization/reconciliation/scientific_review.md`.
- The benchmark-scale source/transpiled P0/P1 study is recorded separately in `artifacts/postfreeze_track_a_transpiled_representation_v1/`; its P1 gate did not justify physical-DAG P2.

