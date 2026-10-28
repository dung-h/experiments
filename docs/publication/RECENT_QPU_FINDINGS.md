# Recent QPU findings — reader guide

Saved-result publication supplement, 2026-10-28. New evidence here does not replace Scientific Freeze v2, Study R, A1, or the frozen grouped Track-A benchmark. The running random-row study has no final result in this package.

## Shots-input ablation

The completed study reused 30 validated no-shots baseline cells and trained 30 matched shots-input cells on the same 4,515 observations, 165 structural groups and five frozen outer folds. Adding fit-standardized raw shots reduced group-balanced MAE by 0.035846 s for MLP and 0.030201 s for Graph+MLP. Their 95% paired structural-group bootstrap intervals are respectively [-0.093131, 0.006669] and [-0.063154, 0.001449] s. Both include zero: no robust benefit was established. The MLP pooled MAE increased slightly, while group-balanced MAE improved; weighting and source/fold heterogeneity must remain visible.

Preprocessing and saved-checkpoint audits PASS in the local receipts. Both architectures worsen on the QPack source slice under group-balanced scoring. Failure diagnostics are post-hoc descriptions of saved predictions, not new fitted explanations or a validated shot-scaling law. Ma–Li/QPack fixed-shot strata and Qonductor variable shots confound shots with source/context.

Reports and aggregate tables: `artifacts/postfreeze_mali_shots_ablation_v1/`. Private per-observation predictions, input tensors and checkpoints are retained locally.

## Analytical interpretation and corrections to older audit prose

The common provider-field lineage is verified: all three sources use provider-recorded `Result.time_taken`, with documented source-specific aggregation and unit conversions. Different source names or contexts do not prove different internal timing boundaries. Exact internal components remain unresolved. The older target-boundary audit's H3 SUPPORTED wording must therefore be read with `docs/audits/qpu_timing_boundary/RESULT_TIME_TAKEN_PROVENANCE_NOTE.md`; it is not evidence that three different provider clocks were measured.

The later implementation audit also qualifies older measurement wording: the saved Qiskit route includes terminal measurements, whereas the QCRE unitary route excludes them. Numerical similarity does not prove identical operation sets. The current source/code and replay evidence is recorded in `docs/audits/notads_hyperq_estimator_20261027/REPORT.md` and `artifacts/postfreeze_notads_hyperq_track_a_v1/REPORT.md`. Existing historical audit files are preserved.

Raw nominal paths, shot-scaled estimates and outer-train affine calibrations remain distinct methods. Calibration learns from observed runtime labels; its improvement does not identify or measure physical overhead. NoTaDS raw scores are abstract units, so no seconds MAE is assigned to them without a declared mapping. HyperQ paper-path replay equals the saved Qiskit route on the present inputs and is not independent evidence or a full HyperQ scheduler reproduction.

Source-stratified correlation and processor-revision slices are descriptive. Generation/source confounding, reconstructed non-Qonductor circuits and nominal rather than execution-day snapshots limit mechanistic claims. IBM formula calculations are local formula applications with stated sub-job assumptions, not retrieved historical remote usage estimates.

## Publication and reproduction boundary

This supplement includes reports, explicitly selected aggregate tables, receipts and producing source. It excludes QPU observation rows, per-row/group error tables, QASM, graph tensors, checkpoints, split memberships and bootstrap identity archives. Source hashes identify retained local inputs; they do not distribute them. Public verification checks file integrity and supplied aggregate arithmetic, not private-row replay or model execution.

See `scripts/export_recent_qpu_findings.py`, `docs/publication/RECENT_QPU_INDEX.json`, and `scripts/verify_recent_qpu_findings.py`. The literature-review plan is a proposed presentation structure with evidence-card requirements, not a completed new survey or an authorization for more training.
