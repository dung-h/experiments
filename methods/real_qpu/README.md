# Real-QPU implementation map

The current comparison assigns 4,515 observations. The
[result table](../../results/real_qpu/method_comparison.md) and
[preprocessing report](../../docs/data_preprocessing.md) are the reader entry
points; these files retain the producing source, not a second public dataset.

| Role | Implementation |
| --- | --- |
| Shared panel, grouped partitions, fresh graph/MLP and regression fits | [run_qualified_logical_benchmark.py](run_qualified_logical_benchmark.py) |
| Hash-checked raw QCRE reuse and analytical completion | [resume_qualified_analytical.py](resume_qualified_analytical.py) |
| Independent saved-evidence metrics and reader export | [finalize_qualified_benchmark.py](finalize_qualified_benchmark.py) |
| Source model and normalization | [mali_model.py](mali_model.py), [mali_full_features.py](mali_full_features.py), [run_mali_batched.py](run_mali_batched.py) |
| Group allocation, regressor candidates and source-QCRE adapter reused by the current runner | [freeze_strict_logical_panel.py](freeze_strict_logical_panel.py), [run_strict_logical_qonductor.py](run_strict_logical_qonductor.py), [run_qonductor_grouped_reproduction.py](run_qonductor_grouped_reproduction.py), [run_strict_logical_analytical.py](run_strict_logical_analytical.py) |
| Source joins, archive-supported recipes and nominal compiled features | `materialize_*`, `prepare_*`, `qonductor_*`, `build_qonductor_logical_recovery.py`, `validate_qonductor_archive_recipes.py` |

Remaining analytical helpers preserve the raw-input derivation used by this
run. Their standalone historical entry points are not additional active
benchmarks. Superseded fitting/reporting drivers have been removed from the
current public tree; originals remain in the research workspace and Git history.

Producing files are byte-identical to the validated source. Their historical
input roots and module names are retained, not silently rewritten as a portable
refit. For example, `run_mali_full_features` maps to the published
`mali_model.py`. [Provenance](../../provenance/real_qpu.json) records these
module/path/hash mappings. Resolve external source observations, circuits,
snapshots, local evidence and environment before invoking them. The reduced
checkout's verification commands never import these experiment runners.
