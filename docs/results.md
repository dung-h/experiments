# Benchmark results

Start with the [Real-QPU reader table](../results/real_qpu/method_comparison.md)
and [simulator reader table](../results/simulator/method_comparison.md). Their
machine-readable sources are the corresponding
[QPU CSV](../results/real_qpu/method_comparison.csv) and
[simulator CSV](../results/simulator/method_comparison.csv); the
[figure manifest](../results/figures/manifest.json) records plotting inputs
and filters. Every score needs its assigned/scored denominator, target clock,
output clock, representation, split and method fidelity.

On the shared 7,350-row QPU panel, graph adaptation MAE/R² is 1.2186 s/0.4291,
matched MLP 1.7322 s/0.1873, and polynomial 1.4006 s/−0.5867. Graph-minus-MLP
MAE is −0.5136 s, with a grouped 95% interval [−0.9074, −0.1748].
Graph-minus-polynomial is −0.1819 s, interval [−0.4883, +0.1968]. The graph
branch adds node/context information as well as topology. Source differences,
large tail errors and reconstruction uncertainty limit generalization.

Simulator scores apply within their engines and clocks. Transpiled Aer GNN
MAE is 0.3008 s against XGBoost 0.2970 s; their paired interval includes zero.
Dense Random Forest has low MAE while work-scaled Ridge has the highest R²
among the evaluated predictors. Joint-MPS family conditioning has no resolved
runtime advantage; predicted-rung quality violations are 64/140. Contraction
and analog companions have their own targets and quality conditions.

[Scientific review](scientific_review.md) explains numerical validation,
uncertainty and limitations. [Paper comparison](paper_comparison.md) connects
these findings to the original mechanisms and targets, with source citations.
[Data preprocessing](data_preprocessing.md) documents selection, reconstruction
and missingness before any interpretation of the scores.
