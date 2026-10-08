# Benchmark results

The real-QPU evaluation dataset contains **4,515 observations**.
Its [method table](../results/real_qpu/method_comparison.md) is pending the
complete fresh-fit run and final QA. Dataset preparation, selection and new
grouped splits are documented in the [dataset report](data_preprocessing.md).
No error scores from another fitted population are reported as current results.

The simulator track retains its independently checked
[method table](../results/simulator/method_comparison.md) and
[CSV](../results/simulator/method_comparison.csv). Comparisons stay within their
engines, clocks, precision and quality conditions.

Transpiled Aer GNN MAE is 0.3008 s against XGBoost 0.2970 s; their paired interval
includes zero. Dense Random Forest has low MAE while work-scaled Ridge has the
highest R² among the evaluated predictors. Joint-MPS family conditioning has
no resolved runtime advantage; predicted-rung quality violations are 64/140.
Contraction and analog companions have their own targets and quality conditions.

[Scientific review](scientific_review.md) explains evidence checks and limits.
[Paper comparison](paper_comparison.md) connects mechanisms to local simulator
findings and distinguishes proposed QPU evaluations from completed results.
