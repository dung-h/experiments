# Slide deck and figures

## Current review deck

The current editable deck is the [14-slide PowerPoint](benchmark_review/quantum_runtime_benchmark_vi.pptx), with a matching [14-page PDF](benchmark_review/quantum_runtime_benchmark_vi.pdf). Its [README](benchmark_review/README.md) records scope, build recipe, and QA. The deck separates archived QPU service-time prediction from local simulator runtime; its Aer and CUDA-Q MPS results are on separate slides and clocks.

The earlier 13-slide export at the top of this directory is retained as a historical snapshot; use `benchmark_review/` for the updated presentation. The [scientific brief](../docs/presentation.md) and [method-by-dataset table](../artifacts/benchmark_v3/results/reader_method_dataset_table.csv) are the reader-facing content and data references.

## Figures

- [Predictor contrasts: Aer and fixed-χ16 MPS](figures_v2/README.md): two new paired-error figures with grouped-bootstrap intervals, denominators, source pins, and reproduction command.
- [Earlier QPU and simulator diagnostics](figures/README.md): four historical charts from the previous slide export.

## Reproducibility boundary

The benchmark figures are rebuilt from checked-in aggregate tables; their script does not train models or run simulators. The editable deck was generated with the Codex Presentations runtime, which is not a normal project dependency and is not installed by the Python requirements file. Rebuild instructions are in [`benchmark_review/README.md`](benchmark_review/README.md). The committed PPTX/PDF are the review deliverables; they are not byte-identical rebuild targets or public-release approval.
